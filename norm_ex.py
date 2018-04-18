from pydub import AudioSegment

# normalizing wav files for testing

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def main():
	for name in ['tammany', 'jake', 'lindsay']:
		sound = AudioSegment.from_file(name+".wav", "wav")
		normalized_sound = match_target_amplitude(sound, -20.0)
		normalized_sound.export(name+"norm.wav", format="wav")

main()