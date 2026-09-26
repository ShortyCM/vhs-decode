import time
from lddecode.core import DemodCache
from vhsdecode.addons.gnuradioZMQ import ZMQSend, ZMQReceive


class DemodCacheTape(DemodCache):
    # VHSRFDecode.demodblock only reads these attributes. In particular, field
    # processing state such as SECAM servo averages, input/output objects, and
    # thread pools must not cross the multiprocessing boundary.
    _WORKER_RF_ATTRIBUTES = (
        "DecoderParams",
        "Filters",
        "_chroma_trap",
        "_disable_diff_demod",
        "_do_cafc",
        "_high_boost",
        "_notch",
        "_options",
        "_sub_emphasis_params",
        "_use_fsc_notch_filter",
        "_video_eq",
        "blockcut",
        "blockcut_end",
        "blocklen",
        "chromaTrap",
        "freq_hz",
    )

    def __init__(self, *args, **kwargs):
        # This must be available while the base class creates its minimal
        # process worker copies.
        self._gnrc_afe = args[0].options.gnrc_afe
        super(DemodCacheTape, self).__init__(*args, **kwargs)
        if self._gnrc_afe:
            print(
                "Open GNURadio with ZMQ REQ source set at tcp://localhost:%d and ZMQ REP sink set at tcp://*:%d"
                % (5555, 5556)
            )
            print(
                "The data stream will be of the float type at 40MSPS (40MHz sample rate)"
            )
            print(
                "It will send the raw RF for further processing prior to demodulation (useful for RF EQ discovery "
                "and group delay compensation)"
            )
            print(
                "You might want to do this in single threaded decode mode (-t 1 parameter) - TODO: might not work correctly with --no_resample yet."
            )

    def _make_worker_copy(self):
        worker = super(DemodCacheTape, self)._make_worker_copy()
        worker._gnrc_afe = self._gnrc_afe
        return worker

    def _make_worker_rf_copy(self):
        """Build a minimal VHSRFDecode shell for the demodulation process."""
        worker_rf = self.rf.__class__.__new__(self.rf.__class__)
        for attribute in self._WORKER_RF_ATTRIBUTES:
            if hasattr(self.rf, attribute):
                setattr(worker_rf, attribute, getattr(self.rf, attribute))

        # Plotting is disabled whenever multiprocessing workers are requested,
        # and plot controller objects need not (and often cannot) be pickled.
        worker_rf.debug_plot = None
        return worker_rf

    def _initialize_worker(self):
        if self._gnrc_afe:
            self.zmqsend = ZMQSend()
            self.zmqreceive = ZMQReceive()

    def worker(self, return_on_empty=False):
        """Override to skip mtf stuff since that's laserdisc specific."""
        blocksrun = 0
        blockstime = 0

        rf = self.rf

        while True:
            if return_on_empty and self.q_in.qsize() == 0:
                return

            item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, _, request = item[1:]

                if self._gnrc_afe:
                    raw_input = block["rawinput"]
                    raw_size = raw_input.size
                    self.zmqsend.send(raw_input)
                    block["rawinput"] = self.zmqreceive.receive(raw_size)

                output = {}

                if "fft" not in block:
                    fftdata = None
                else:
                    fftdata = block["fft"]

                st = time.time()
                output["demod"] = rf.demodblock(
                    data=block["rawinput"],
                    fftdata=fftdata,
                    mtf_level=0,
                    cut=True,
                )
                blockstime += time.time() - st
                blocksrun += 1

                output["request"] = request
                output["MTF"] = 0  # Not used so just set to 0 for time.

                self.q_out.put((blocknum, output))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])
