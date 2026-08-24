"""A JAX reimplementation of the ell1fit log-posterior, for gradient samplers.

Why this exists
---------------
:mod:`ell1fit.posterior` builds a log-posterior out of numba kernels. That is
the right shape for a sampler that only ever asks for values -- emcee -- but
NUTS asks for *gradients*, and nothing in the numba path can supply them. This
module rebuilds the same posterior out of JAX primitives so that
``jax.grad`` works on it. :func:`run_nuts` is what ``--sampler nuts`` calls;
``tools/sampler_bench.py`` puts the same rebuild through its own adapter so a
gradient sampler can be measured on the same footing as the ensemble ones.

Nothing here is imported at package load time -- every function that needs
``jax`` or ``numpyro`` imports them lazily, so the base ``ell1fit`` install
does not require either. ``run_nuts`` raises a clear ``ImportError`` pointing
at ``pip install ell1fit[nuts]`` if they are missing.

It is a **reimplementation, not a wrapper**, and that is the risk it carries:
two expressions of one model can drift apart silently. Two things hold them
together. :func:`check_against_numba` compares the two log-posteriors at fixed
probe positions, and :func:`check_gradient` compares this module's analytic
gradient against finite differences of the *numba* posterior -- so the gradient
is checked against the original implementation rather than against itself.
:func:`run_nuts` runs both before sampling, and both are cheap.

What is deliberately different
------------------------------
**The deorbiting loop runs a fixed number of iterations.** The numba kernel
iterates until successive estimates agree to ``tolerance`` (1e-8 s). A
data-dependent trip count is not what a reverse-mode autodiff wants, and the
loop does not need one: the map contracts by the projected orbital velocity in
units of c, around 1e-3 for a real binary, so each pass gains three decimal
digits and a dozen passes are far past where float64 stops changing. See
:data:`DEORBIT_ITERATIONS`.

**Hard prior bounds return ``-inf`` with a zero gradient.** A NUTS trajectory
that steps outside a flat prior's support gets infinite energy and the step is
rejected as divergent, which is what the bound means. It is not a smooth
barrier and does not pretend to be.
"""

import numpy as np


#: Fixed-point passes in the deorbiting iteration. The numba kernel stops on a
#: 1e-8 s change instead, which is not a rule reverse-mode autodiff can follow.
#:
#: Six, because that is where float64 stops changing rather than where a
#: tolerance is met. The circular first guess is wrong by ``A1 * e``, about
#: 0.04 s here, and each pass multiplies that by the projected orbital velocity
#: in units of c -- 6.4e-4 for this binary -- so three passes already reach
#: 1e-11 s. Measured on P1, the log-posterior is **bitwise identical** at 6 and
#: at 12 passes (105.02545962633513 both times) and first moves in the eleventh
#: digit at 4. Six therefore buys a full safety factor over what is needed and
#: still costs half of twelve: value and gradient together fall from 3351 us to
#: 1713 us, against 575 us for one numba value evaluation. That halving is the
#: whole reason this constant is not simply set large and forgotten -- the
#: backward pass replays every iteration, so each one is paid twice.
DEORBIT_ITERATIONS = 6

#: Density floor before the logarithm, matching ``UniformCubicTemplate.loglike``.
TEMPLATE_FLOOR = 1e-12


def enable_x64():
    """Put JAX in float64 before anything builds an array.

    Not optional. The dominant phase term is ``F0 * t``, which reaches ~1e6
    cycles here; float32 would leave ~0.1 cycles of resolution, which is a
    hundred times coarser than the whole posterior is wide.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


def _second_order_coefficients(jnp, EPS1, EPS2):
    """JAX twin of :func:`ell1fit.phase_utils._second_order_coefficients`."""
    e1e1, e2e2, e1e2 = EPS1 * EPS1, EPS2 * EPS2, EPS1 * EPS2
    return (
        1.0 - (3.0 * e1e1 + 5.0 * e2e2) / 8.0,
        e1e2 / 4.0,
        EPS2 / 2.0,
        -EPS1 / 2.0,
        -3.0 * (e1e1 - e2e2) / 8.0,
        -3.0 * e1e2 / 4.0,
    )


def _ell1_shape(sin_phase, cos_phase, a1s, a1c, a2s, a2c, a3s, a3c):
    """JAX twin of :func:`ell1fit.phase_utils._ell1_shape`.

    Written with the same multiple-angle identities and in the same order, so
    that any disagreement between the two implementations is a difference in
    floating-point association rather than in the expression itself.
    """
    ss = sin_phase * sin_phase
    cc = cos_phase * cos_phase
    return (
        a1s * sin_phase
        + a1c * cos_phase
        + a2s * (2.0 * sin_phase * cos_phase)
        + a2c * (1.0 - 2.0 * ss)
        + a3s * (sin_phase * (3.0 - 4.0 * ss))
        + a3c * (cos_phase * (4.0 * cc - 3.0))
    )


def _deorbit(jnp, lax, times, pb, a1, tasc, eps1, eps2):
    """Remove ELL1 orbital delays by a fixed number of fixed-point passes.

    Mirrors :func:`ell1fit.phase_utils._ell1_deorbit`: the same circular first
    guess, then the same second-order update. The only change is the stopping
    rule -- see :data:`DEORBIT_ITERATIONS`.
    """
    omega = 2.0 * jnp.pi / pb
    t = times - tasc
    coefficients = _second_order_coefficients(jnp, eps1, eps2)

    def body(_, out):
        phase = omega * out
        return t - a1 * _ell1_shape(jnp.sin(phase), jnp.cos(phase), *coefficients)

    # The circular first guess, identical to the numba kernel's.
    out = t - a1 * jnp.sin(omega * t)
    out = lax.fori_loop(0, DEORBIT_ITERATIONS, body, out)
    return out + tasc


def _fast_phase(jnp, times, frequency_derivatives):
    """JAX twin of :func:`ell1fit.phase_utils._fast_phase_generic`.

    The specialised one- and two-derivative kernels in ``phase_utils`` exist to
    save an array multiply per term under numba; XLA fuses the whole expression
    either way, so only the general form is carried here. It is the same Taylor
    series accumulated in the same order.
    """
    total = jnp.zeros_like(times)
    t_pow = jnp.ones_like(times)
    fact = 1.0
    n = 0.0
    for f in frequency_derivatives:
        t_pow = t_pow * times
        n += 1
        fact *= n
        total = total + (1.0 / fact * f) * t_pow
    return total


def _template_terms(jnp, phases, coefficients, x0, dx, n_intervals):
    """Evaluate the uniform-grid cubic template at each phase.

    JAX twin of :func:`ell1fit.templates._evaluate_uniform_cubic_floored`. The
    interval index is a piecewise constant of phase, so it contributes no
    gradient -- correct, because the spline is continuous across its knots and
    the derivative comes entirely from the polynomial piece.
    """
    ph = phases - jnp.floor(phases)
    # ``x0`` is -dx/2, so ``(ph - x0) / dx`` is never negative and the numba
    # kernel's truncating ``int()`` and this floor agree.
    j = jnp.clip(jnp.floor((ph - x0) / dx).astype(jnp.int32), 0, n_intervals - 1)
    u = ph - (x0 + j * dx)
    c = coefficients[j]
    return c[:, 0] + u * (c[:, 1] + u * (c[:, 2] + u * c[:, 3]))


def _template_loglike(jnp, phases, coefficients, x0, dx, n_intervals, weights):
    """Summed log-density of ``phases``, weighted or not.

    Follows :meth:`ell1fit.templates.UniformCubicTemplate.loglike`: the floor is
    applied to whatever goes inside the logarithm, which for a weighted fit is
    the mixture ``1 + w (T - 1)`` rather than the template itself.
    """
    values = _template_terms(jnp, phases, coefficients, x0, dx, n_intervals)
    if weights is not None:
        values = weights * values + (1.0 - weights)
    return jnp.sum(jnp.log(jnp.maximum(values, TEMPLATE_FLOOR)))


def _prior_parameters(func):
    """Read a prior closure's constants by name.

    The priors in :mod:`ell1fit.priors` are closures, so their constants live in
    free variables rather than attributes. Pulling them out by name is only as
    stable as those names, which is exactly why every translated prior is then
    checked numerically against the original -- see :func:`_translate_prior`.
    """
    cells = func.__closure__ or ()
    return dict(zip(func.__code__.co_freevars, (cell.cell_contents for cell in cells)))


def _check_prior_translation(func, translated, probe_values):
    """Assert a translated prior reproduces the original at every probe value.

    Deliberately a separate function rather than a block at the end of
    :func:`_translate_prior`. The translated priors are closures over their
    constants, and a closure captures the *variable*, not its value -- so a
    check sharing their scope can rebind a name they close over and change the
    function it just finished approving. That is not hypothetical: this check
    once used a local called ``scale``, evaluated the rebuilt prior, passed,
    and then clobbered the Gaussian's width with an unrelated number. Every
    Gaussian prior went flat and the checker reported success, because it had
    already taken its samples. Keeping the scopes disjoint makes that
    unrepresentable.
    """
    reference = np.array([float(func(v)) for v in probe_values])
    rebuilt = np.array([float(translated(v)) for v in probe_values])
    if not np.array_equal(np.isfinite(reference), np.isfinite(rebuilt)):
        raise AssertionError(f"Translated prior {func!r} disagrees on where it is finite")
    finite = np.isfinite(reference) & np.isfinite(rebuilt)
    if finite.any():
        worst = np.max(np.abs(reference[finite] - rebuilt[finite]))
        magnitude = max(1.0, np.max(np.abs(reference[finite])))
        if worst > 1e-10 * magnitude:
            raise AssertionError(f"Translated prior {func!r} differs by {worst:.3e}")


def _translate_prior(jnp, func, probe_values):
    """Rebuild one log-prior as a JAX expression, and verify the rebuild.

    ``probe_values`` are physical values spanning the range the parameter can
    plausibly reach. The translated prior must reproduce the original at every
    one of them, including the infinities, or this raises: a prior that is
    quietly wrong changes the answer without changing the diagnostics.
    """
    qualname = getattr(func, "__qualname__", "")
    constants = _prior_parameters(func)

    if qualname.startswith("_flat_logprior"):
        low, high = constants["bound0"], constants["bound1"]

        def translated(value):
            return jnp.where((value < low) | (value > high), -jnp.inf, 0.0)

    elif qualname.startswith("_periodic_uniform_logprior"):
        center = constants["center"]
        period = constants["period"]
        half_width = constants["half_width"]

        def translated(value):
            dx = jnp.mod(value - center + 0.5 * period, period) - 0.5 * period
            return jnp.where(jnp.abs(dx) > half_width, -jnp.inf, 0.0)

    elif qualname.startswith("_periodic_normal_logprior"):
        center = constants["center"]
        period = constants["period"]
        sigma = constants["sigma"]
        norm_const = constants["norm_const"]

        def translated(value):
            dx = jnp.mod(value - center + 0.5 * period, period) - 0.5 * period
            return norm_const - 0.5 * (dx / sigma) ** 2

    elif hasattr(func, "__self__") and hasattr(func.__self__, "kwds"):
        # A frozen ``scipy.stats.norm``; ``assign_logpriors`` builds these with
        # keyword arguments only.
        frozen = func.__self__
        loc = float(frozen.kwds.get("loc", 0.0))
        scale = float(frozen.kwds.get("scale", 1.0))
        norm_const = -0.5 * np.log(2 * np.pi) - np.log(scale)

        def translated(value):
            return norm_const - 0.5 * ((value - loc) / scale) ** 2

    else:
        raise NotImplementedError(
            f"No JAX translation for log-prior {qualname!r}. Add one in "
            "ell1fit/nuts_sampling.py rather than letting the sampler run "
            "against a prior it does not implement."
        )

    _check_prior_translation(func, translated, probe_values)
    return translated


def _frequency_derivative_names(parameters, file_index):
    """Names of the frequency derivatives present for one file, in order."""
    names = []
    count = 0
    while f"F{count}_{file_index}" in parameters:
        names.append(f"F{count}_{file_index}")
        count += 1
    return names


def build_jax_logpost(observations, setup, jit=True):
    """Build the log-posterior of ``setup`` as a differentiable JAX function.

    Takes the same position vector as
    :func:`ell1fit.posterior._build_posterior_functions`'s third return value --
    local coordinates, ``physical = local * factor + initial`` -- so a position
    is interchangeable between the two implementations and the harness can
    compare them directly.

    Returns
    -------
    callable
        ``position -> log posterior``, differentiable with ``jax.grad``.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    from .templates import UniformCubicTemplate

    parameters = setup.parameters
    names = list(setup.parameter_names)
    factors = [float(f) for f in setup.factors]
    baseline = [float(v) for v in setup.baseline_values]
    index_of = {name: k for k, name in enumerate(names)}
    n_files = len(observations.times_from_pepoch)

    times = [jnp.asarray(t, dtype=jnp.float64) for t in observations.times_from_pepoch]
    zero = jnp.zeros(1, dtype=jnp.float64)

    templates = []
    for template in setup.template_funcs:
        if not isinstance(template, UniformCubicTemplate):
            raise NotImplementedError(
                f"Only UniformCubicTemplate is translated to JAX, got {type(template).__name__}"
            )
        templates.append(
            (
                jnp.asarray(template.coefficients, dtype=jnp.float64),
                float(template.x0),
                float(template.dx),
                int(template.n_intervals),
            )
        )

    weights = None
    if setup.weights is not None:
        weights = [jnp.asarray(w, dtype=jnp.float64) for w in setup.weights]

    # Priors are probed over the interval the local coordinate would have to
    # cross to leave a sane region: +-50 sigma in local units, which for a hard
    # bound straddles it and for a Gaussian spans its tails.
    priors = []
    for k, func in enumerate(setup.logprior_funcs):
        probe = baseline[k] + factors[k] * np.linspace(-50.0, 50.0, 201)
        priors.append(_translate_prior(jnp, func, probe))

    frequency_names = [_frequency_derivative_names(parameters, i) for i in range(n_files)]

    def physical(name, position):
        """Physical value of ``name``: from the position if free, else fixed."""
        k = index_of.get(name)
        if k is None:
            return float(parameters[name])
        return position[k] * factors[k] + baseline[k]

    def logpost(position):
        logprior = 0.0
        for k, prior in enumerate(priors):
            logprior = logprior + prior(position[k] * factors[k] + baseline[k])

        pb = physical("PB", position)
        a1 = physical("A1", position)
        eps1 = physical("EPS1", position)
        eps2 = physical("EPS2", position)
        tasc = physical("TASC", position)

        loglike = 0.0
        for i in range(n_files):
            pb_i = pb + parameters.get(f"PB_offset_{i}", 0.0)
            a1_i = a1 + parameters.get(f"A1_offset_{i}", 0.0)
            eps1_i = eps1 + parameters.get(f"EPS1_offset_{i}", 0.0)
            eps2_i = eps2 + parameters.get(f"EPS2_offset_{i}", 0.0)

            tasc_raw = (
                tasc + parameters.get(f"TASC_offset_{i}", 0.0) - parameters[f"PEPOCH_{i}"]
            ) * 86400.0
            tasc_i = jnp.mod(tasc_raw + 0.5 * pb_i, pb_i) - 0.5 * pb_i

            deorbited = _deorbit(jnp, lax, times[i], pb_i, a1_i, tasc_i, eps1_i, eps2_i)
            deorbited_pepoch = _deorbit(jnp, lax, zero, pb_i, a1_i, tasc_i, eps1_i, eps2_i)

            derivatives = [physical(name, position) for name in frequency_names[i]]
            phase_pepoch = _fast_phase(jnp, deorbited_pepoch, derivatives)[0]
            phases = (
                physical(f"Phase_{i}", position)
                - phase_pepoch
                + _fast_phase(jnp, deorbited, derivatives)
            )
            # Wrapped here and again inside the template, exactly as the numba
            # path does. Not redundant: a phase a hair below zero wraps to
            # exactly 1.0 on the first pass and to 0.0 on the second.
            phases = phases - jnp.floor(phases)

            coefficients, x0, dx, n_intervals = templates[i]
            loglike = loglike + _template_loglike(
                jnp,
                phases,
                coefficients,
                x0,
                dx,
                n_intervals,
                None if weights is None else weights[i],
            )

        # The same screen ``_calculate_phases`` applies, which raises there and
        # is turned into -inf by ``func_to_maximize``. Expressed on the global
        # orbital values, as it is there.
        velocity_over_c = jnp.abs(a1) * 2 * jnp.pi / jnp.abs(pb)
        invertible = velocity_over_c * (1.0 + jnp.abs(eps1) + jnp.abs(eps2)) < 1.0

        return jnp.where(
            jnp.isinf(logprior), logprior, jnp.where(invertible, loglike + logprior, -jnp.inf)
        )

    return jax.jit(logpost) if jit else logpost


def check_against_numba(observations, setup, start, logpost, n_probes=24, jitter=None, seed=90210):
    """Check the JAX rebuild reproduces the numba posterior it stands in for.

    Probes at the starting point and at ``n_probes`` positions jittered around
    it, on the scale the posterior is actually sampled on rather than a scale of
    order one -- in local coordinates one unit is a million sigma, so a probe
    ball of radius one lands outside every prior and would compare two ways of
    saying ``-inf``.

    Parameters
    ----------
    logpost : callable
        The numba posterior being stood in for -- ``func_to_maximize`` from
        :func:`ell1fit.posterior._build_posterior_functions`.

    Returns
    -------
    dict
        ``worst_absolute_difference`` alongside the range the log-posterior
        covers over the same probes, which is what makes the first number
        readable: a discrepancy matters relative to how much the posterior
        varies across the region being sampled, not in absolute terms.

    Notes
    -----
    Exact agreement is not on offer and its absence is not a defect. The numba
    deorbiting kernels are compiled with ``fastmath``, so their arithmetic is
    reassociated; the two implementations differ in the last bit of the
    deorbited times, around 1.5e-11 s on a 1e5 s baseline, and a few times 1e4
    events accumulate that into ~1e-6 of log-posterior. Tightening the numba
    iteration's tolerance does not move it, which is what identifies it as
    rounding rather than an unconverged loop.
    """
    from .scaling import TARGET_LOCAL_SIGMA

    if jitter is None:
        jitter = TARGET_LOCAL_SIGMA
    ndim = len(start)
    rng = np.random.default_rng(seed)
    probes = np.vstack([start, start + jitter * rng.normal(size=(n_probes, ndim))])

    logpost_jax = build_jax_logpost(observations, setup)
    reference = np.array([float(logpost(p)) for p in probes])
    rebuilt = np.array([float(logpost_jax(p)) for p in probes])
    return {
        "n_probes": int(probes.shape[0]),
        "worst_absolute_difference": float(np.max(np.abs(reference - rebuilt))),
        "logpost_range_over_probes": float(reference.max() - reference.min()),
    }


def check_gradient(observations, setup, start, n_probes=4, jitter=None, seed=90211):
    """Check ``jax.grad`` of the rebuilt posterior against central differences.

    Differenced against **this module's own** log-posterior rather than the
    numba one, and deliberately so. Differencing the numba posterior sounds
    like the stronger test but is not available: its value carries ~1e-6 of
    rounding (see :func:`check_against_numba`), which a step of a hundredth of
    a sigma amplifies into a few times 1e-4 relative -- so such a check reports
    the differencing error and would pass whether or not the gradient were
    right. The two checks divide the work instead: this one says the derivative
    matches the function, :func:`check_against_numba` says the function matches
    the model.
    """
    import jax

    from .scaling import TARGET_LOCAL_SIGMA

    if jitter is None:
        jitter = TARGET_LOCAL_SIGMA
    ndim = len(start)
    rng = np.random.default_rng(seed)
    probes = np.vstack([start, start + jitter * rng.normal(size=(n_probes, ndim))])

    logpost_jax = build_jax_logpost(observations, setup)
    gradient = jax.jit(jax.grad(logpost_jax))
    step = 0.01 * jitter

    worst = 0.0
    for probe in probes:
        analytic = np.asarray(gradient(probe))
        differenced = np.empty_like(analytic)
        for k in range(ndim):
            ahead, behind = probe.copy(), probe.copy()
            ahead[k] += step
            behind[k] -= step
            differenced[k] = (float(logpost_jax(ahead)) - float(logpost_jax(behind))) / (2 * step)
        floor = 1e-6 * np.max(np.abs(analytic))
        worst = max(
            worst, np.max(np.abs(analytic - differenced) / np.maximum(np.abs(differenced), floor))
        )
    return {"n_probes": int(probes.shape[0]), "worst_relative_difference": float(worst)}


def run_nuts(
    observations,
    setup,
    func_to_maximize,
    starting_pars,
    outroot="chain_results",
    labels=None,
    corner_labels=None,
    draws=4000,
    chains=4,
    target_accept=0.8,
    max_tree_depth=10,
    rhat_gate=1.01,
    seed=None,
):
    """Run NUTS (numpyro) against the JAX rebuild, and summarize like emcee.

    The production counterpart of :func:`ell1fit.mcmc_utils.safe_run_sampler`,
    called when ``optimize_solution(..., sampler="nuts")``. Two things it does
    *not* do, both worth knowing before reaching for it:

    - **No checkpointing.** ``safe_run_sampler`` resumes from an HDF5 backend
      because an ensemble chain can need hundreds of thousands of steps.
      NUTS needs far fewer -- 202 ESS/s against 69.5 on the benchmark's small
      problem -- so a single run is the v1 design; there is no equivalent of
      ``outroot + '.h5'`` here yet.
    - **No adaptive re-run.** ``safe_run_sampler`` keeps sampling until its
      autocorrelation-based convergence check passes or ``max_n`` is spent.
      This runs exactly ``draws`` per chain, once, and reports whatever R-hat
      that reached -- logged as a warning, not retried, if it is above
      ``rhat_gate``.

    ``draws`` is total per chain, split evenly between warm-up and kept
    samples -- the same convention ``tools/sampler_bench.py`` uses, and not
    the same quantity as ``--nsteps`` (which sizes the emcee ensemble and is
    tuned for its much larger calls-per-effective-sample).

    Before sampling, :func:`check_against_numba` and :func:`check_gradient`
    verify this JAX rebuild against the numba posterior it stands in for;
    their results are logged and returned as ``jax_agreement`` /
    ``jax_gradient_check`` so a silent drift between the two implementations
    is visible in the output rather than only in a log file.

    Returns
    -------
    dict
        Same shape as :func:`ell1fit.mcmc_utils.calculate_result_array_from_samples`
        (``label_p`` percentiles, ``date``, ``nsamples``), plus ``rhat_max``,
        ``divergences``, and the two verification results above.
    """
    import logging

    try:
        import jax
        import jax.numpy as jnp
        from numpyro.diagnostics import summary
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError(
            "sampler='nuts' needs jax and numpyro: pip install ell1fit[nuts]"
        ) from exc

    from .mcmc_utils import plot_mcmc_results

    starting_pars = np.asarray(starting_pars)
    ndim = len(starting_pars)
    if labels is None:
        labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))

    enable_x64()
    agreement = check_against_numba(observations, setup, starting_pars, func_to_maximize)
    derivative = check_gradient(observations, setup, starting_pars)
    logging.info(f"JAX/numba agreement check: {agreement}")
    logging.info(f"JAX gradient check: {derivative}")

    logpost = build_jax_logpost(observations, setup)

    def potential(position):
        return -logpost(position)

    seed = np.random.default_rng().integers(2**31) if seed is None else seed
    rng = np.random.default_rng(seed)
    start = starting_pars + rng.normal(0.0, 1e-6, size=(chains, ndim))

    warmup = draws // 2
    kept = draws - warmup
    kernel = NUTS(
        potential_fn=potential,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=kept,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    key = jax.random.PRNGKey(seed)
    mcmc.run(key, init_params=jnp.asarray(start), extra_fields=("diverging",))
    chain_samples = np.asarray(mcmc.get_samples(group_by_chain=True))
    diverging = np.asarray(mcmc.get_extra_fields()["diverging"])

    rhat = summary(chain_samples)["Param:0"]["r_hat"]
    rhat_max = float(np.max(rhat))
    if rhat_max >= rhat_gate:
        logging.warning(
            f"NUTS R-hat {rhat_max:.4f} did not clear the gate ({rhat_gate}) after "
            f"{draws} draws on {chains} chains; the posterior below may be under-sampled."
        )

    flat_samples = chain_samples.reshape(-1, ndim)
    result_dict = {}
    percs = [1, 10, 16, 50, 84, 90, 99]
    for i in range(ndim):
        mcmc_percentiles = np.percentile(flat_samples[:, i], percs)
        for i_p, p in enumerate(percs):
            result_dict[labels[i] + f"_{p:g}"] = mcmc_percentiles[i_p]

    from astropy.time import Time

    result_dict["date"] = Time.now().mjd
    result_dict["nsamples"] = flat_samples.shape[0]
    result_dict["rhat_max"] = rhat_max
    result_dict["divergences"] = int(np.sum(diverging))
    result_dict["jax_agreement"] = agreement
    result_dict["jax_gradient_check"] = derivative

    plot_mcmc_results(
        flat_samples=flat_samples,
        labels=corner_labels or labels,
        fname=outroot + "_corner.jpg",
    )

    return result_dict
