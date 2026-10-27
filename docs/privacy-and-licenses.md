# Privacy and licenses

* No network code exists in this crate. Audio, settings and features never leave the machine.
* No external crates and no models are used, so there are no third-party licences to track yet.
* If a learned aligner is added later: record model/code licences, size, source and offline-loading procedure here.
* WAV files are opened read-only. `oto.ini` is only replaced after backup and verification; `.otoium-state.tsv` stores
  the lines otoium generated so manual edits can be detected.
