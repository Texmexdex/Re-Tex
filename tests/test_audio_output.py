"""Output device picking: full names (DirectSound) open through MME, which cuts names at 31 characters."""
from mashlab.player import Player

OUTS = [(13, "MME", "Microsoft Sound Mapper - Output"), (14, "MME", "Realtek Digital Output (Realtek"),
        (18, "MME", "Speakers (Logitech PRO X Wirele"), (19, "MME", "CABLE Input (VB-Audio Virtual C"),
        (20, "MME", "CABLE In 16ch (VB-Audio Virtual"),
        (37, "Windows DirectSound", "Primary Sound Driver"),
        (38, "Windows DirectSound", "Realtek Digital Output (Realtek(R) Audio)"),
        (42, "Windows DirectSound", "Speakers (Logitech PRO X Wireless Gaming Headset)"),
        (43, "Windows DirectSound", "CABLE Input (VB-Audio Virtual Cable)"),
        (44, "Windows DirectSound", "CABLE In 16ch (VB-Audio Virtual Cable)"),
        (99, "Windows WASAPI", "Only On WASAPI Device")]


def test_names_and_resolve(monkeypatch):
    monkeypatch.setattr(Player, "_outputs", staticmethod(lambda: OUTS))
    assert Player.output_names() == ["Realtek Digital Output (Realtek(R) Audio)",
                                     "Speakers (Logitech PRO X Wireless Gaming Headset)",
                                     "CABLE Input (VB-Audio Virtual Cable)", "CABLE In 16ch (VB-Audio Virtual Cable)"]
    assert Player._resolve("Speakers (Logitech PRO X Wireless Gaming Headset)") == 18   # MME, by its cut name
    assert Player._resolve("CABLE In 16ch (VB-Audio Virtual Cable)") == 20
    assert Player._resolve("Only On WASAPI Device") == 99                             # no MME entry: exact name
    assert Player._resolve("Unplugged Headset") is None                              # -> system default
    assert Player._resolve(None) is None
