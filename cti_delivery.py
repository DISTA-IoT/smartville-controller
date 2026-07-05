import random


class CTIDeliveryModel:
    """
    Models imperfect CTI delivery (reviewer comment RC 3.4).

    The base TIGER game assumes that, upon payment, cyber-threat intelligence is
    delivered as clean, correctly labelled, complete traffic traces that become
    usable instantly: taking the epistemic action moves a G2 class straight into
    the Knowns set, and its replay buffer -- already holding true-labelled online
    samples -- becomes eligible for inference-module training in the very same
    tick (see NewTigerEnvironment.perform_epistemic_action and
    TigerBrain.sample_from_replay_buffers, which skips a class's buffer while it
    is still a G2). Real CTI feeds are not this generous: label quality varies,
    captures are partial, and intelligence arrives with delay.

    This model reintroduces those three imperfections as orthogonal, composable
    configuration channels. EVERY channel defaults to the clean regime, so an
    unconfigured model is a strict no-op (`enabled` is False) and all existing
    experiments reproduce byte-for-byte:

      - label noise      (cti_label_noise, [0,1]):     a fraction of the acquired
                          class's training-label draws are relabelled to a wrong
                          class, poisoning its prototype.
      - partial capture  (cti_capture_fraction, (0,1]): only this fraction of the
                          acquired class's samples are actually admitted into its
                          replay buffer, so the prototype is built from sparser,
                          noisier evidence.
      - delivery delay   (cti_delivery_delay, steps):  the class becomes trainable
                          (G2 -> Known) only `delay` decision steps after payment;
                          during the lag it stays an unlearnable, still-costly
                          zero-day. Optional +/- `cti_delivery_jitter` randomises
                          the lag.

    Coupling (cti_quality_coupling = 'none' | 'purity' | 'confidence'): when set,
    the label-noise and partial-capture severities scale with the (hidden)
    quality of the cluster the CTI was bought for. Buying intelligence for a
    messy, low-purity / low-confidence cluster yields noisier, sparser data than
    buying for a clean one. This is the lever that makes a state-conditioned (RL)
    policy genuinely necessary: blind "buy-early" / "buy-periodically" /
    "buy-when-uncertain" heuristics cannot separate a good purchase from a
    poisonous one, whereas a policy that reads the IM's cluster-quality signal
    can learn to buy selectively.

    The delivery delay is deliberately NOT coupled to cluster quality: it models
    feed/vendor latency, which is independent of how clean a given sample is.

    All state (pending queue, active corruption) is per-episode and cleared by
    reset(); a dedicated seeded RNG keeps the delay/jitter draws reproducible.
    """

    def __init__(self, id_kwargs):
        get = id_kwargs.get
        self.label_noise = float(get('cti_label_noise', 0.0) or 0.0)
        self.label_noise_mode = str(get('cti_label_noise_mode', 'symmetric') or 'symmetric')
        self.capture_fraction = float(get('cti_capture_fraction', 1.0) or 1.0)
        self.delivery_delay = int(get('cti_delivery_delay', 0) or 0)
        self.delivery_jitter = int(get('cti_delivery_jitter', 0) or 0)
        self.quality_coupling = str(get('cti_quality_coupling', 'none') or 'none')

        # Per-episode state (see reset()).
        # _pending: purchases paid for but not yet delivered (delivery delay).
        #   Each entry is {'label', 'deliver_step', 'noise', 'capture'}.
        # _active:  labels already delivered and now corrupted for the rest of
        #   the episode: label -> {'noise', 'capture'}.
        self._pending = []
        self._active = {}
        self._rng = random.Random()

    @property
    def enabled(self):
        """True iff any channel departs from the clean/instant/complete regime."""
        return (self.label_noise > 0.0 or self.capture_fraction < 1.0
                or self.delivery_delay > 0 or self.delivery_jitter > 0)

    def reset(self, seed):
        """Clear all per-episode delivery state and reseed the delay RNG."""
        self._pending = []
        self._active = {}
        self._rng = random.Random(seed)

    def _severity(self, purity, confidence):
        """
        Coupling multiplier in [0,1] applied to the label-noise and capture
        knobs: 0 => pristine CTI (a perfectly pure/confident cluster), 1 =>
        worst-case (the configured knob at full strength). With coupling 'none'
        (or a missing signal) the knobs apply flat, at full severity.
        """
        if self.quality_coupling == 'purity' and purity is not None:
            return max(0.0, min(1.0, 1.0 - float(purity)))
        if self.quality_coupling == 'confidence' and confidence is not None:
            return max(0.0, min(1.0, 1.0 - float(confidence)))
        return 1.0

    def on_purchase(self, label, current_step, purity=None, confidence=None):
        """
        Register a CTI purchase of `label` made at `current_step`. Computes this
        purchase's effective corruption from the configured knobs and the
        cluster-quality coupling, schedules delivery `delay` steps out, and
        returns (delay, noise, capture).

        When delay == 0 the corruption is registered active immediately (instant
        delivery) and the caller performs the G2 -> Known move in the same tick;
        when delay > 0 the purchase is queued and pop_ready() delivers it later.
        """
        sev = self._severity(purity, confidence)
        noise = self.label_noise * sev
        capture = 1.0 - (1.0 - self.capture_fraction) * sev

        delay = self.delivery_delay
        if self.delivery_jitter > 0:
            delay += self._rng.randint(-self.delivery_jitter, self.delivery_jitter)
        delay = max(0, delay)

        if delay == 0:
            self._active[label] = {'noise': noise, 'capture': capture}
        else:
            self._pending.append({'label': label,
                                  'deliver_step': current_step + delay,
                                  'noise': noise, 'capture': capture})
        return delay, noise, capture

    def has_pending(self):
        return bool(self._pending)

    def pending_labels(self):
        """Labels paid for but not yet delivered -- excluded from re-purchase."""
        return {p['label'] for p in self._pending}

    def pop_ready(self, current_step):
        """
        Move every pending delivery whose deliver_step <= current_step into the
        active (corrupted) set and return their labels -- the classes that
        become trainable this tick. Returns [] when nothing is due.
        """
        if not self._pending:
            return []
        ready, still = [], []
        for p in self._pending:
            if p['deliver_step'] <= current_step:
                self._active[p['label']] = {'noise': p['noise'], 'capture': p['capture']}
                ready.append(p['label'])
            else:
                still.append(p)
        self._pending = still
        return ready

    def noise_for(self, label):
        """Effective label-noise fraction for a delivered class (0.0 if clean)."""
        entry = self._active.get(label)
        return entry['noise'] if entry else 0.0

    def capture_for(self, label):
        """Effective capture fraction for a delivered class (1.0 if complete)."""
        entry = self._active.get(label)
        return entry['capture'] if entry else 1.0
