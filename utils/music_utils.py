import numpy as np
import music21

# Scale intervals
INTERVALS_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Ionian mode
INTERVALS_MINOR_SCALE_NATURAL = [0, 2, 3, 5, 7, 8, 10]  # Aeolian mode
INTERVALS_MINOR_SCALE_HARMONIC = [0, 2, 3, 5, 7, 8, 11]
INTERVALS_MINOR_SCALE_MELODIC = [0, 2, 3, 5, 7, 9, 11]

# Keys in an octave
OCTAVE_KEY_LIST = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


def normalize_equivalent_pitch(pitch_name):
    """
    This is based on the equal temperament. (Of course)
    Also, change flat notes' names to X-
    """
    if pitch_name in ["Cs", "C#"]:
        return "D-"
    elif pitch_name in ["Ds", "D#"]:
        return "E-"
    elif pitch_name in ["Es", "E#"]:
        return "F"
    elif pitch_name in ["Fs", "F#"]:
        return "G-"
    elif pitch_name in ["Gs", "G#"]:
        return "A-"
    elif pitch_name in ["As", "A#"]:
        return "B-"
    elif pitch_name in ["Bs", "B#"]:
        return "C"
    elif pitch_name in ["Db", "Eb", "Gb", "Ab", "Bb"]:
        return pitch_name[0] + "-"
    else:
        return pitch_name


def midi_number_to_pitch(midi_number):
    """
    Convert a midi number to its pitch. The octave will also be returned.
    """
    pitch = OCTAVE_KEY_LIST[(midi_number - 60) % 12]
    octave = (midi_number - 60) // 12
    return pitch, octave


def chord_name_to_pitch(chord_name):
    """
    Convert a chord name to a list of pitches without octaves.
    """
    chord = music21.harmony.ChordSymbol(chord_name)
    pitches = [pitch.name for pitch in chord.pitches]

    return pitches  # [p.midi for p in chord.pitches]
