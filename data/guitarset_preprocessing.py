# Author: Andy Wiggins <awiggins@drexel.edu>
# Functions for preprocessing guitarset data

from math import ceil
from globals import *
import numpy as np
import scipy
from mirdata.annotations import F0Data
from util import crop_or_pad, chunk_arr
import librosa
import crepe

def crepe_compute_f0(audio, f0data, sr=SR, frame_rate=FRAME_RATE, conf_thresh=CREPE_CONFIDENCE_THRESH):
    """
    Takes in an audio array and computes f0 using the off-the-shelf CREPE pitch tracker. The data in f0data is overwritten by CREPE's predictions.

    Parameters
    ----------
    audio : numpy array (samples)
    f0data : F0Data
        Dict from mirdata containing the times, frequencies and voicings
        These will be overwritten by crepe's predictions
    sr : float
        sampling rate
    frame_rate : float
        desired frames per second of f0 annotation
    conf_thresh : float in [0,1]
        confidence threshold. freqs below the threshold are zeroed out

    Returns
    ----------
    f0data : mirdata F0Data dict
        contains times, frequencies, and voicing predicted by crepe
    """
    step_size = int(round(1000 / frame_rate)) # milliseconds per frame

    if abs(step_size - (1000 / frame_rate)) > EPS:
        print("Error! invalid frame rate. Crepe requires an integer step size (in milliseconds).")

    # returns times, frequencies, confidences, and model output acitivations
    times, freqs, confs, activs = crepe.predict(audio, sr=sr, step_size=step_size, viterbi=True, verbose=False)

    freqs[np.where(confs <= conf_thresh)] = 0.0

    # crop or pad freqs and times to match the length of the voicing extracted
    freqs = crop_or_pad(freqs, size=len(f0data.voicing))

    f0data.frequencies = freqs

    f0data.voicing[np.where(freqs == 0.0)] = 0.0
    
    return f0data


def extract_loudness(audios, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT, desired_len=None, scale_to_MIDI_range=False):
    """
    Takes in an array of audio signals and stft params and computes a frame-wise loudness in db using A_weighting. Implementation partially borrowed from ddsp.

    Parameters
    ----------
    audios : numpy array (batch, samples)
        2D numpy array of audio signals
    sr : float
        sampling rate
    hop_length : int
        hop length for stft computation (determines frame rate)
    n_fft : int
        number of points in fft
    desired_len : int
        The number of frames we want (I'm getting one more frame than the annotations, this will force all to be the same shape)
    scale_to_MIDI_range : bool
        If true, output will be between 0 and 127
        Default is false, where the output is unscaled

    Returns
    ----------
    loudness : numpy array
        2D array of loudnesses with shape (batch, frames)
    """

    # compute log spectrogram
    power_spec = np.array([np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop_length)) ** 2 for a in audios])

    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    np.seterr(divide = 'ignore')  # suppress divide error which is for some reason caused by getting a_weighting
    a_weighting = librosa.A_weighting(fft_freqs)[np.newaxis, :, np.newaxis] # insert a new axes for broadcasting across, batches and frames
    np.seterr(divide = 'ignore') # turn divide errors back on 

    # perform weighting in linear scale, a_weighting given in decibels
    weighting = librosa.db_to_power(a_weighting)
    power_spec = power_spec * weighting

    # average over frequencies (weighted power per a bin)
    avg_power = np.mean(power_spec, axis=1) # axis 1 is freq bins
    # make sure to crop or pad before computing db,since I don't want to pad zeros on db loudness
    if desired_len is not None:
        avg_power = crop_or_pad(avg_power, desired_len)
    loudness = librosa.power_to_db(avg_power)

    if scale_to_MIDI_range:
        gset_loudness_range = GSET_MAX_STRING_LOUDNESS_BOUND - GSET_MIN_STRING_LOUDNESS_BOUND
        loudness -= GSET_MIN_STRING_LOUDNESS_BOUND # adds 40, for a min bound of -40 for instance
        loudness = np.clip(loudness, 0, gset_loudness_range) # clip to [0, gset_loudness_range]
        loudness = (loudness / gset_loudness_range) * 127 # scale to [0, 127]

    return loudness



def resample_labels(f0data, orig_frame_rate=GSET_FRAME_RATE, target_frame_rate=FRAME_RATE):
    """
    Takes in a mirdata F0Data object and resamples the times, freqs and voicings, for a given frame rate. Portions adapted from mirdata codebase.

    Parameters
    ----------
    f0data : F0Data
        Dict from mirdata containing the times, frequencies and voicings
    orig_frame_rate : float
        original frame rate
    target_frame_rate : float
        desired frame rate

    Returns
    ----------
    resampled_f0data : F0Data
        dict of updated F0 data
    """
    times = f0data.times
    frequencies = f0data.frequencies
    voicing = f0data.voicing

    annotation_duration = (times[1] - times[0]) * len(times)
    times_idx = np.arange(annotation_duration * target_frame_rate)
    frame_duration = 1 / target_frame_rate
    times_new = times_idx * frame_duration

    # Fix zero transitions by replacing zero values with last nonzero value
    # avoids errors in resampling
    frequencies_held = np.array(frequencies)
    for n, frequency in enumerate(frequencies[1:]):
        if frequency == 0:
            frequencies_held[n + 1] = frequencies_held[n]
    # Linearly interpolate frequencies
    frequencies_resampled = scipy.interpolate.interp1d(times, frequencies_held, "linear", bounds_error=False, fill_value=0.0)(times_new)
    # Retain zeros
    frequency_mask = scipy.interpolate.interp1d(times, frequencies, "zero", bounds_error=False, fill_value=0)(times_new)
    frequencies_resampled *= frequency_mask != 0

    voicing_resampled = scipy.interpolate.interp1d(times, voicing, "nearest", bounds_error=False, fill_value=0)(times_new)

    voicing_resampled[frequencies_resampled == 0] = 0

    return F0Data(times_new, 
                f0data.time_unit, 
                frequencies_resampled, 
                f0data.frequency_unit, 
                voicing_resampled, 
                f0data.voicing_unit)

def make_audio_match_annotation_duration(audio, f0data, sr=SR):
    """
    Takes in an audio array and a mirdata F0Data object and crops the audio to match the length of the annotation. Trims the silence at the end of an audio clip when the string is done playing. 

    Parameters
    ----------
    audio : numpy array (sampled at SR)
        audio to be cropped
    f0data : F0Data
        Dict from mirdata containing the times, frequencies and voicings.
    sr : float
        sampling rate of the audio

    Returns
    ----------
    updated_audio : numpy array (sampled at SR)
        audio, sliced/padded to length corresponding to the time of the annotations 
    """

    times = f0data.times
    if times[0] != 0:
        print("Error. This annotation starts at a time other than 0 seconds. crop_audio_to_annotation function needs to be updated to handle this case.")
        exit(0)
    annotation_duration = (times[1] - times[0]) * len(times) # seconds

    desired_samples = int(round(annotation_duration * sr))
    updated_audio = crop_or_pad(audio, desired_samples)

    return updated_audio

def trim_beginning(audio, f0data, sr=SR, beginning_frames=16):
    """
    Takes in an audio array and a mirdata F0Data object and trims the beginning of each until the first non-zero voicing. 

    Parameters
    ----------
    audio : numpy array (sampled at SR)
        audio to be trimmed
    f0data : F0Data
        Dict from mirdata containing the times, frequencies and voicings.
    sr : float
        sampling rate of the audio
    beginning_frames : int
        number of frames before the first annotation

    Returns
    ----------
    updated_audio : numpy array (sampled at SR)
        audio, with beginning silence trimmer
    updated_f0data : F0Data
        F0data annotation, trimmed before first  
    """
    times = f0data.times
    frequencies = f0data.frequencies
    voicing = f0data.voicing
    
    frame_rate = 1 / (times[1] - times[0])
    start_frame = 0
    for i, voi in enumerate(voicing):
        if voi != 0:
            start_frame = i
            break
    # start several frames before the first voicing if possible
    new_start_frame = max(0, start_frame - beginning_frames)

    times = times[new_start_frame:]
    frequencies = frequencies[new_start_frame:]
    voicing = voicing[new_start_frame:]
    #update times
    shifted_times = times - times[0] # subtract the time at the new frame zero to shift times

    new_start_sample = int(round(new_start_frame * (sr / frame_rate)))
    udpated_audio = audio[new_start_sample:]

    updated_f0data = F0Data(shifted_times, 
                f0data.time_unit, 
                frequencies, 
                f0data.frequency_unit, 
                voicing, 
                f0data.voicing_unit)

    return udpated_audio, updated_f0data

def chunk_audio_and_annotation(audio, f0data, item_dur=ITEM_DUR, sr=SR, frame_rate=FRAME_RATE):
    """
    Takes in an audio array and a mirdata F0Data object and splits them in chunks of a given duration. (Pads the final chunk if necessary.) 

    Parameters
    ----------
    audio : numpy array (sampled at SR)
        audio to be chunked
    f0data : F0Data
        mirdata F0Data containing the times, frequencies and voicings to be chunked
    item_dur : float
        time in seconds for chunks (to be fed into network)
    sr : float
        sampling rate of the audio
    frame_rate : float
        frame rate of the annotations  

    Returns
    ----------
    audio_arr : 2D numpy array 
        chunked audio array of shape (num_chunks, chunk_size). In samples. 
    freq_arr : 2D numpy array
        chunked frequency array of shape (num_chunks, chunk_size). In frames. 
    voicing_arr : 2D numpy array
        chunked voicing array of shape (num_chunks, chunk_size). In frames. 

    """
    audio_chunk_size = item_dur * sr
    anno_chunk_size = item_dur * frame_rate
    audio_arr = chunk_arr(audio, audio_chunk_size)

    frequencies = f0data.frequencies
    voicing = f0data.voicing
    freq_arr = chunk_arr(frequencies, anno_chunk_size)
    voicing_arr = chunk_arr(voicing, anno_chunk_size)

    return audio_arr, freq_arr, voicing_arr

def chunk_audio(audio, item_dur=ITEM_DUR, sr=SR):
    """
    Takes in an audio array and splits it into chunks of a given duration. 

    Parameters
    ----------
    audio : numpy array (samples)
        audio to be chunked
    item_dur : float
        time in seconds for chunks (to be fed into network)
    sr : float
        sampling rate of the audio

    Returns
    ----------
    audio_arr : 2D numpy array 
        chunked audio array of shape (num_chunks, chunk_size). In samples. 

    """
    audio_chunk_size = item_dur * sr
    audio_arr = chunk_arr(audio, audio_chunk_size)

    return audio_arr

def chunk_frame_labels(labels, item_dur=ITEM_DUR, frame_rate=FRAME_RATE):
    """
    Takes in an array of frame-rate labels and splits it into chunks of a given duration. 

    Parameters
    ----------
    labels : numpy array (frames)
        frame-rate labels to be chunked
    item_dur : float 
        duration of items
    frame_rate : float 
        frames per sec

    Returns
    ----------
    label_arr : 2D numpy array 
        chunked audio array of shape (num_chunks, chunk_size). In frames. 

    """
    label_chunk_size = item_dur * frame_rate
    label_arr = chunk_arr(labels, label_chunk_size)

    return label_arr

def remove_items_with_zero_voicing(dataset):
    """
    Looks for indices corresponding to items with voicing=0 for the entire duration. Removes these elements from all provided arrays) 

    Parameters
    ----------
    dataset : dict of np arrays
        Dataset to be cleaned. (Must have voicing) 

    Returns
    ----------
    dataset : dict of np arrays
        Cleaned dataset.
    """
    indices = []
    for i, v in enumerate(dataset["voicing"]):
        if np.sum(v) <= 0.0:
            indices.append(i)
    for key, arr in dataset.items():
        dataset[key] = np.delete(arr, indices, axis=0)
    
    return dataset
        

    




