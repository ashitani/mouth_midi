from mouth_midi import *

port_index = 0              # MIDI output port number
cc_channels = [1]           # MIDI CC channel
sensitivity = 2.5           # sensitivity from mouth size to CC value
polarity = 1                # MIDI polarity
mosaic_size = 0             # if 0, eye mosaic is disabled
record_file = None          # if None, recording is disabled

sender = MouthMidiSender(
    port_index, cc_channels, sensitivity, polarity, mosaic_size, record_file)
sender.loop()
