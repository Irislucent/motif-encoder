import logging
import numpy as np
import pretty_midi


def get_midi_tempo(filename, first_only=True):
    """
    Get the midi tempo in a midi file.

    Args:
        filename (str):     Path to the input .mid file.
        first_only (bool):  if True, only return the first tempo value, regardless of whether there're tempo changes.
                            if False, return the whole tempo list.

    Return:
        tempo (float):      Tempo of the midi, in beats per minute.
    """

    midi_data = pretty_midi.PrettyMIDI(filename)
    tempo_change_times, tempos = midi_data.get_tempo_changes()
    if first_only:
        tempo = tempos[0]
        return tempo
    else:
        return tempos


def read_midi(filename, index=0):
    """
    Read a single-track MIDI file into a note matrix. Only one instrument is selected.

    Args:
        filename (str):     Path to the input .mid file.
        index (int):        The index of instrument to read.

    Return:
        notes (np.array):   The note event matrix. Each row is like [onset(sec), offset(sec), pitch(MIDInumber), velocity]
                            Its shape is (n, 4), where n is the number of notes.
    """

    midi_data = pretty_midi.PrettyMIDI(filename, resolution=440)
    if len(midi_data.instruments) == 0:
        return np.empty((0, 4))
    instrument = midi_data.instruments[index]

    n_note = len(instrument.notes)
    notes = np.zeros((n_note, 4))

    for i in range(n_note):
        note_event = instrument.notes[i]
        notes[i] = (
            note_event.start,
            note_event.end,
            note_event.pitch,
            note_event.velocity,
        )

    idx = np.argsort(notes[:, 0])
    notes = notes[idx, ...]

    return notes


def read_midi_multitrack(filename):
    """
    Read a multi_track MIDI file into a list of note matrices.

    Args:
        filename (str):     Path to the input .mid file.
        index (int):        The index of instrument to read.

    Return:
        tracks (list)   :   Each element in the list is a note event matrix. Each row is like [onset(sec), offset(sec),
                            pitch(MIDInumber), velocity]. Its shape is (n, 4), where n is the number of notes.
    """

    midi_data = pretty_midi.PrettyMIDI(filename, resolution=440)
    tracks = []
    for i in range(len(midi_data.instruments)):
        instrument = midi_data.instruments[i]

        n_note = len(instrument.notes)
        notes = np.zeros((n_note, 4))

        for j in range(n_note):
            note_event = instrument.notes[j]
            notes[j] = (
                note_event.start,
                note_event.end,
                note_event.pitch,
                note_event.velocity,
            )

        idx = np.argsort(notes[:, 0])
        notes = notes[idx, ...]

        tracks.append(notes)
    return tracks


def read_midi_multitrack_unify_tempo(filename):
    """
    Read a multi_track MIDI file into a list of note matrices. Normalize the tempo so that temporal information is represented using beats only.

    Args:
        filename (str):     Path to the input .mid file.

    Return:
        tracks (list)   :   Each element in the list is a note event matrix. Each row is like [onset(beats), offset(beats),
                            pitch(MIDInumber), velocity]. Its shape is (n, 4), where n is the number of notes.
    """

    midi_data = pretty_midi.PrettyMIDI(filename, resolution=440)
    tempo_change_times, tempos = midi_data.get_tempo_changes()
    tracks = []
    for i in range(len(midi_data.instruments)):
        instrument = midi_data.instruments[i]

        n_note = len(instrument.notes)
        notes = np.zeros((n_note, 4))

        if len(tempos) == 1:
            tempo = tempos[0]
            for j in range(n_note):
                note_event = instrument.notes[j]
                notes[j] = (
                    note_event.start,
                    note_event.end,
                    note_event.pitch,
                    note_event.velocity,
                )
                notes[j][0] = notes[j][0] * tempo / 60
                notes[j][1] = notes[j][1] * tempo / 60
        else:
            for j in range(n_note):
                note_event = instrument.notes[j]
                notes[j] = (
                    note_event.start,
                    note_event.end,
                    note_event.pitch,
                    note_event.velocity,
                )
                note_event_new = np.zeros(2)
                # onset
                # check in which section
                for k in range(len(tempos)):
                    if notes[j][0] < tempo_change_times[k]:
                        onset_section_id = k
                        break
                    if k == len(tempos) - 1:
                        onset_section_id = k + 1
                # compute time in beats, first compute passed sections
                for k in range(onset_section_id):
                    if k > 0:
                        note_event_new[0] += (
                            (tempo_change_times[k] - tempo_change_times[k - 1])
                            * tempos[k - 1]
                            / 60
                        )  # how many beats in this tempo section?
                # then compute the section where it is
                note_event_new[0] += (
                    (notes[j][0] - tempo_change_times[onset_section_id - 1])
                    * tempos[onset_section_id - 1]
                    / 60
                )

                # offset
                # check in which section
                for k in range(len(tempos)):
                    if notes[j][1] < tempo_change_times[k]:
                        offset_section_id = k
                        break
                    if k == len(tempos) - 1:
                        offset_section_id = k + 1
                # compute time in beats, first compute passed sections
                for k in range(offset_section_id):
                    if k > 0:
                        note_event_new[1] += (
                            (tempo_change_times[k] - tempo_change_times[k - 1])
                            * tempos[k - 1]
                            / 60
                        )  # how many beats in this tempo section?
                # then compute the section where it is
                note_event_new[1] += (
                    (notes[j][1] - tempo_change_times[offset_section_id - 1])
                    * tempos[offset_section_id - 1]
                    / 60
                )

                notes[j][:2] = note_event_new[:2]

        idx = np.argsort(notes[:, 0])
        notes = notes[idx, ...]

        tracks.append(notes)
    return tracks


def read_midi_to_pianoroll(filename, index=0, grid=1/32, pianoroll_len=None, binary=True, compress_84=True):
    """
    Read a single-track MIDI file into a pianoroll matrix. Only one instrument is selected.

    Args:
        filename (str)      :   Path to the input .mid file.
        index (int)         :   The index of instrument to read.
        grid (float)        :   The time resolution of the pianoroll. Default is 1/32th note.
        pianoroll_len (int) :   Len of grids in the whole pianoroll. Might be trimmed or padded.
        binary (bool)        :   If True, all velocities are disabled and pianorollis binary.
        compress_84 (bool)   :   If True, compress the 128-dim pianoroll to 84-dim by removing the first 24 and last 20 pitches.

    Return:
        pianoroll (np.array):   The pianoroll matrix.
    """
    midi_data = pretty_midi.PrettyMIDI(filename, resolution=440)
    midi_tempo = get_midi_tempo(filename)

    if len(midi_data.instruments) == 0:
        raise Exception("This midi has no instruments.")
    instrument = midi_data.instruments[index]
    fs = 1 / grid / 4 * (midi_tempo / 60)
    pianoroll = instrument.get_piano_roll(fs=fs)

    if pianoroll_len is not None:
        if pianoroll.shape[1] > pianoroll_len:
            pianoroll = pianoroll[:, :pianoroll_len]
        elif pianoroll.shape[1] < pianoroll_len:
            new_pianoroll = np.zeros((128, pianoroll_len))
            new_pianoroll[:, :pianoroll.shape[1]] = pianoroll
            pianoroll = new_pianoroll

    if binary:
        pianoroll = np.where(pianoroll > 0, 1, 0)

    if compress_84:
        pianoroll = pianoroll[24:108, :]

    return pianoroll


def seconds_to_beats(notes, tempo):
    """
    Convert time elements in a note matrix from seconds to beats.
    """
    for i, note in enumerate(notes):
        notes[i][:2] *= tempo / 60
    return notes


def beats_to_seconds(notes, tempo):
    """
    Convert time elements in a note matrix from beats to seconds.
    """
    for i, note in enumerate(notes):
        notes[i][:2] *= 60 / tempo
    return notes


def write_midi(filename, notes, bpm=60, is_drum=False):

    """
    Write a note matrix into a MIDI file. Only used for single-track midi.

    Args:
        notes (np.array):   The note event matrix. Each row is like [onset(sec), offset(sec), pitch(MIDInumber), velocity]
                            Its shape is (n, 4), where n is the number of notes.
        bpm (int):          Tempo used to initialize PrettyMIDI object.
        is_drum (bool):     Whether this midi track is a drum track.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=440)
    instr = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )
    instr.is_drum = is_drum

    for note in notes:
        note = pretty_midi.Note(
            velocity=int(note[3]), pitch=int(note[2]), start=note[0], end=note[1]
        )
        instr.notes.append(note)

    midi.instruments.append(instr)
    midi.write(filename)


def tempo2sec(bpm):
    return 60.0 / bpm
