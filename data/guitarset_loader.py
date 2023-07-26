# Author: Andy Wiggins <awiggins@drexel.edu>
# Class for loading GuitarSet and creating datasets from it

import mirdata
import librosa
import numpy as np
from globals import *
import data.guitarset_preprocessing as gset_proc
import os
from tqdm import tqdm, tqdm_notebook
from util import crop_or_pad

class GuitarSetLoader:
	"""
	Class for loading GuitarSet and creating datasets from it.
	"""
	def __init__(self, data_home=GSET_PATH, download=False, use_crepe_f0_labels=USE_CREPE_F0_LABELS):
		"""
		Initialize parameters for loading guitarset.
		
		Parameters
		----------
		data_home : string
			Path to the directory containing the GuitarSet dataset, e.g. '~/Desktop/Datasets/Dataset'
		download : bool
			True if the dataset needs to be downloaded
		use_crepe_f0_labels : bool
			True if the monophonic f0 labels should be recomputed using the pretrained crepe model
		"""
		self.guitarset = mirdata.initialize('guitarset', data_home=data_home)
		if download:
			self.guitarset.download()
		self.tracks = self.guitarset.load_tracks()

		self.use_crepe_f0_labels = use_crepe_f0_labels
		
	def split_dataset_by_player_id(self, test_player_id=0):
		"""
		Splits GuitarSet dataset by player id with one player held out for testing.

        Parameters
        ----------
        test_player_id : int
        	ID # of player to hold split out for testing

        Returns
        ----------
        splits : dict
        	Dictionary where keys are split names, the values are lists of the track_ids
        """
		if type(test_player_id) == type(3):
			test_player_id = "0" + str(test_player_id)
			
		if not test_player_id in ["00", "01", "02", "03", "04", "05"]:
			print("Invalid GuitarSet player_id...defaulting to splitting out test player 00.")
			test_player_id = "00"
			
		splits = {"test_player-" + test_player_id: [], 
		"train_player-not" + test_player_id: []}
		
		for track_id in self.tracks.keys():
			if self.tracks[track_id].player_id == test_player_id:
				splits["test_player-" + test_player_id].append(track_id)
			else:
				splits["train_player-not" + test_player_id].append(track_id)
		
		return splits

	def filter_splits_by_version(self, splits, version=None):
		"""
		Given a dictionary of dataset splits, filters to keep only files of a certain version: soloing or comping. 
		Note: In mirdata this version is also called 'mode'.
		Also updates the split names to include the version

        Parameters
        ----------
		splits : dict
        	Dictionary where keys are split names, the values are lists of the track_ids
        version : string
        	"solo" or "comp" to include only solo tracks or comping tracks
			if None, will do nothing

        Returns
        ----------
        filtered_splits : dict
        	splits_dict, now processed to include only tracks of the desired version
        """
		# function to check a track_ID's version
		is_version = lambda ID, v: self.tracks[ID].mode == v

		if version is None:
			filtered_splits = splits

		elif version not in ["solo", "comp"]:
			print("Invalid GuitarSet track version. Must be in[\"comp\", \"solo\"]")
			print("Not filtering for version.")
			filtered_splits = splits

		else:
			filtered_splits = {}
			for split, track_IDs in splits.items():				
				filtered_IDs = [ID for ID in track_IDs if is_version(ID, version)]
				updated_split_name = f"{split}_{version}" 
				filtered_splits[updated_split_name] = filtered_IDs

		return filtered_splits


	def get_audio_and_MIDI_conditioning_from_track(self, track_id, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR, omit_incomplete_chunks=OMIT_INCOMPLETE_CHUNKS):
		"""
		Given a track ID, item duration (and sampling and frame rates), compute a conditioning array inspired by the work in:
		https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf. Conditioning contains information about note onsets and pitches.

        Parameters
        ----------
		track_id : string
        	ID # of track
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset
		omit_incomplete_chunks : bool
        	if True, use the floor to only have only complete chunks 
			In other words, get rid of the final item from tracks if incomplete and would hard cut to zero

        Returns
        ----------
		track_data : dict
        	track_data['conditioning'] : np array (items, frames, num_strings=6, pitch_or_onset=2)
				midi pitch and onset velocity for each string
			track_data['mic_audio'] : np array (items, samples)
				audio from room microphone
			track_data['mix_audio'] : np array (items, samples)
				audio mixture summed from hexaphonic signal		
        """

		# load track from ID
		track = self.tracks[track_id]

		# create empty dict for all string data
		track_data = {}

		# load mix, mic and hex audios
		mix_audio, mix_sr = track.audio_mix
		if mix_sr != sr:
			mix_audio = librosa.resample(mix_audio, orig_sr=mix_sr, target_sr=sr)
		mic_audio, mic_sr = track.audio_mic
		if mic_sr != sr:
			mic_audio = librosa.resample(mic_audio, orig_sr=mic_sr, target_sr=sr)

		# chunk audio
		mix_audio_arr = gset_proc.chunk_audio(mix_audio, item_dur=item_dur, sr=sr)
		mic_audio_arr = gset_proc.chunk_audio(mic_audio, item_dur=item_dur, sr=sr)

		# set the number of desired items and frames
		if omit_incomplete_chunks:
			track_items = int(np.floor(len(mix_audio) / (sr * item_dur)))
		else:
			track_items = int(np.ceil(len(mix_audio) / (sr * item_dur)))
		frames_per_item = int(round( item_dur * frame_rate ))
		frames_per_track = track_items * frames_per_item

		# create empty conditioning array 
		conditioning = np.zeros((track_items, frames_per_item, 6, 2)) # (items, frames, strings, pitch/onset_velocity)	

		# load hex audio
		audio_hex, hex_sr = track.audio_hex_cln

		for string_idx, string_let in enumerate(GUITAR_STRING_LETTERS):

			# load the individual string audio, resampling if necessary
			string_audio = audio_hex[string_idx]
			if hex_sr != sr:
				string_audio = librosa.resample(string_audio, orig_sr=hex_sr, target_sr=sr)

			### compute the midi pitch array
					
			# get the string's pitch contour
			string_pitch_contour = track.pitch_contours[string_let]		

			# if no pitch labels for the string, move onto the next string
			if string_pitch_contour == None:
				continue

			# resample pitch contour to desired frame rate
			string_pitch_contour = gset_proc.resample_labels(string_pitch_contour, target_frame_rate=frame_rate)

			# optionally replace pitch contours with predictions from crepe
			if self.use_crepe_f0_labels:
				string_pitch_contour = gset_proc.crepe_compute_f0(string_audio, string_pitch_contour, sr=sr, frame_rate=frame_rate)

			# pad freqs with zero to be an appropriate length
			freqs = crop_or_pad(string_pitch_contour.frequencies, size=frames_per_track)
			# convert from hz to midi, clip between 0 and 127
			midi_pitches = np.clip(librosa.hz_to_midi(freqs), 0, 127)
			# chunk the midi pitches into items
			midi_pitch_arr = gset_proc.chunk_frame_labels(midi_pitches, item_dur=item_dur, frame_rate=frame_rate) # (items, frames)


			# compute framewise loudness of the string audio, expand to get batch dimension
			# make sure to set scale to MIDI-range to true to get loudness in [0, 127]
			string_loudness = gset_proc.extract_loudness(np.expand_dims(string_audio, axis=0), 
														sr=sr, 
														hop_length=int(round(sr/frame_rate)), 
														desired_len=frames_per_track, 
														scale_to_MIDI_range=True)[0]			
			
			# initialize stringwise onset velocity 
			onset_vel = np.zeros((track_items * frames_per_item)) # shape: (total frames)

			# get just the onset times of all the notes on the given string
			onset_times = track.notes[string_let].intervals[:,0] 
			# convert time to frames
			onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length=int(round(sr/frame_rate)))

			# filter out frames that happen out of bounds (in the final item that's being omitted)
			onset_frames = [frm for frm in onset_frames if (frm >= 0 and frm < len(onset_vel) )]

			# get the onset velocities from the string loudness
			onset_vel[onset_frames] = string_loudness[onset_frames]

			# chunk onset_vel
			onset_vel_arr = gset_proc.chunk_frame_labels(onset_vel, item_dur=item_dur, frame_rate=frame_rate) # (items, frames)
			
			# store conditioning
			conditioning[:,:,string_idx,0] = midi_pitch_arr
			conditioning[:,:,string_idx,1] = onset_vel_arr

		
		# store track data in dict
		track_data['conditioning'] = conditioning # shape : (items, frames, num_strings=6, pitch_or_onset=2)
		track_data['mic_audio'] = mic_audio_arr # shape : (items, samples)
		track_data['mix_audio'] = mix_audio_arr # shape : (items, samples)
		
		return track_data


	#### Methods for original DDSP-style labels (F0, loudness) ####

	def get_1_string_audios_and_annos_from_track(self, track_id, string_let, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a track ID, string and duration (and sampling and frame rates), compute arrays of audio, freqs, voicing, and loudness. If no activity on the string, return None for all.

        Parameters
        ----------
		track_id : string
        	ID # of track
        string_let : string
        	guitar string to extract for.  Must be in ["E", "A", "D", "G", "B", "e"]
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset


        Returns
        ----------
        audio_arr : numpy array (items, samples)
        	Cleaned audios of single string from track
		freq_arr : numpy array (items, frames)
        	F0 annotations of single string from track
			Either from gset or crepe depending on self.use_crepe_f0_labels
		voicing_arr : numpy array (items, frames)
        	Binary voicings of single string from track
		loudness_arr : numpy array (items, frames)
        	Loudnesses of single string from track
        """

		track = self.tracks[track_id]
		string_pitch_contour = track.pitch_contours[string_let]

		if string_pitch_contour is None:
			return None, None, None, None # return None for all if no activity on the string

		audio_hex, orig_sr = track.audio_hex_cln
		string_audio = audio_hex[GUITAR_STRING_LETTERS.index(string_let)]
		if orig_sr != sr:
			string_audio = librosa.resample(string_audio, orig_sr=orig_sr, target_sr=sr)

		
		# if not using crepe, gset labels need to be resampled and audio length adjusted to annotation duration
		string_pitch_contour = gset_proc.resample_labels(string_pitch_contour, target_frame_rate=frame_rate)
		string_audio = gset_proc.make_audio_match_annotation_duration(string_audio, string_pitch_contour)

		if self.use_crepe_f0_labels:
			string_pitch_contour = gset_proc.crepe_compute_f0(string_audio, string_pitch_contour, sr=sr, frame_rate=frame_rate)
		
		string_audio, string_pitch_contour = gset_proc.trim_beginning(string_audio, string_pitch_contour, sr=sr, beginning_frames=16)

		audio_arr, freq_arr, voicing_arr = gset_proc.chunk_audio_and_annotation(string_audio, string_pitch_contour, item_dur=item_dur, sr=sr, frame_rate=FRAME_RATE)
		loudness_arr = gset_proc.extract_loudness(audio_arr, sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=freq_arr.shape[1])
		
		return audio_arr, freq_arr, voicing_arr, loudness_arr

	def get_6_string_data_from_track(self, track_id, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a track ID and duration (and sampling and frame rates), compute a dict of arrays of audio, freqs, voicing, and loudness by string.

        Parameters
        ----------
		track_id : string
        	ID # of track
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset

        Returns
        ----------
		track_data: dict of dicts
			['E_audio_arr'] : numpy array (items, samples)
				Cleaned isolated audios of E string from track
			['E_f0_arr'] : numpy array (items, frames)
				F0 annotations of E string from track
				from GuitarSet labels unless self.use_crepe_f0-labels is true
			['E_voicing_arr'] : numpy array (items, frames)
				Binary voicings of E string from track
			['E_loudness_arr'] : numpy array (items, frames)
				Loudnesses of single string from track
			['A...'] (same keys as above)
			['D...'] (same keys as above)
			['G...'] (same keys as above)
			['B...'] (same keys as above)
			['e...'] (same keys as above)
			['mix_audio_arr'] : mix of the monophonic string audio (items, samples)
			['mic_audio_arr'] : audio from the room mic (items, samples)
			['mix_loudness_arr'] : loudness in dB of the mix audio (items, frames)
			['mic_loudness_arr'] : loudness in dB of the mic audio (items, samples)
        """

		# load track from ID
		track = self.tracks[track_id]

		# create empty dict for all string data
		track_data = {}
	
		# load mix, mic and hex audios
		mix_audio, mix_sr = track.audio_mix
		if mix_sr != sr:
			mix_audio = librosa.resample(mix_audio, orig_sr=mix_sr, target_sr=sr)
		mic_audio, mic_sr = track.audio_mic
		if mic_sr != sr:
			mic_audio = librosa.resample(mic_audio, orig_sr=mic_sr, target_sr=sr)
		
		mix_audio_arr = gset_proc.chunk_audio(mix_audio, item_dur=item_dur, sr=sr)
		mic_audio_arr = gset_proc.chunk_audio(mic_audio, item_dur=item_dur, sr=sr)

		# set the number of desired items and frames
		track_items = int(np.ceil(len(mix_audio) / (sr * item_dur)))
		frames_per_item = int(round( item_dur * frame_rate ))
		frames_per_track = track_items * frames_per_item

		mix_loudness_arr = gset_proc.extract_loudness(mix_audio_arr, sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=frames_per_item)
		mic_loudness_arr = gset_proc.extract_loudness(mic_audio_arr, sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=frames_per_item)

		track_data['mix_audio_arr'] = mix_audio_arr
		track_data['mic_audio_arr'] = mic_audio_arr
		track_data['mix_loudness_arr'] = mix_loudness_arr
		track_data['mic_loudness_arr'] = mic_loudness_arr


		# load hex audio
		audio_hex, hex_sr = track.audio_hex_cln

		for string_let in GUITAR_STRING_LETTERS:
			
			string_audio = audio_hex[GUITAR_STRING_LETTERS.index(string_let)]
			if hex_sr != sr:
				string_audio = librosa.resample(string_audio, orig_sr=hex_sr, target_sr=sr)
			
			string_pitch_contour = track.pitch_contours[string_let]

			# if no pitch labels for the string, put all 0s
			if string_pitch_contour == None:
				f0_arr = np.zeros((track_items, frames_per_item))
				voicing_arr = np.zeros((track_items, frames_per_item))

			else:
				# resample pitch contour to desired frame rate
				string_pitch_contour = gset_proc.resample_labels(string_pitch_contour, target_frame_rate=frame_rate)
				if self.use_crepe_f0_labels: # overwrite f0 label with crepe labels
					string_crepe_pitch_contour = gset_proc.crepe_compute_f0(string_audio, string_pitch_contour, sr=sr, frame_rate=frame_rate)

				# pad freqs and voicings with zero to be an appropriate length
				freqs = crop_or_pad(string_pitch_contour.frequencies, size=frames_per_track)
				voicing = crop_or_pad(string_pitch_contour.voicing, size=frames_per_track)
				
				f0_arr = gset_proc.chunk_frame_labels(freqs, item_dur=item_dur, frame_rate=frame_rate)
				voicing_arr = gset_proc.chunk_frame_labels(voicing, item_dur=item_dur, frame_rate=frame_rate)

			audio_arr = gset_proc.chunk_audio(string_audio, item_dur=item_dur, sr=sr)
			loudness_arr = gset_proc.extract_loudness(audio_arr, sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=frames_per_item)
		
			track_data[f"{string_let}_audio_arr"] = audio_arr
			track_data[f"{string_let}_f0_arr"] = f0_arr
			track_data[f"{string_let}_voicing_arr"] = voicing_arr
			track_data[f"{string_let}_loudness_arr"] = loudness_arr

		return track_data

	def create_1_string_dataset(self, track_ids, string_let, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a list of track IDs, string and duration, compute arrays of audio, freqs, voicing, and loudness 

        Parameters
        ----------
		track_ids : list of strings
        	ID # of tracks
		string_let : string
        	guitar string to extract for.  Must be in ["E", "A", "D", "G", "B", "e"]
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset


        Returns
        ----------
        dataset : dict of numpy arrays
        	['audio'] : (items, samples)
			['f0'] : (items, frames)
			['voicing'] : (items, frames)
			['loudness'] : (items, frames)
        """
		dataset = {}

		audios = []
		freqs = []
		voicings = []
		loudnesses = []

		for id in tqdm_notebook(track_ids):
			audio_arr, freq_arr, voicing_arr, loudness_arr = self.get_1_string_audios_and_annos_from_track(id, string_let, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			if audio_arr is not None:
				audios.append(audio_arr)
				freqs.append(freq_arr)
				voicings.append(voicing_arr)
				loudnesses.append(loudness_arr)

		audios = np.concatenate(audios)
		freqs = np.concatenate(freqs)
		voicings = np.concatenate(voicings)
		loudnesses = np.concatenate(loudnesses)

		dataset['audio'] = audios
		dataset['f0'] = freqs
		dataset['voicing'] = voicings
		dataset['loudness'] = loudnesses

		dataset = gset_proc.remove_items_with_zero_voicing(dataset) # comment out for now, as it's causing trouble

		return dataset
	
	def create_6_string_dataset(self, track_ids, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a list of track IDs, sr, frame rate and item duration, compute audio and label arrays for a full 6 string dataset

        Parameters
        ----------
		track_ids : list of strings
        	ID # of tracks
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset

        dataset: dict of dicts
			['E_audio_arr'] : numpy array (items, samples)
				Cleaned isolated audios of E string from track
			['E_f0_arr'] : numpy array (items, frames)
				F0 annotations of E string from track
				from GuitarSet labels unless self.use_crepe_f0-labels is true
			['E_voicing_arr'] : numpy array (items, frames)
				Binary voicings of E string from track
			['E_loudness_arr'] : numpy array (items, frames)
				Loudnesses of single string from track
			['A...'] (same keys as above)
			['D...'] (same keys as above)
			['G...'] (same keys as above)
			['B...'] (same keys as above)
			['e...'] (same keys as above)
			['mix_audio_arr'] : mix of the monophonic string audio (items, samples)
			['mic_audio_arr'] : audio from the room mic (items, samples)
			['mix_loudness_arr'] : loudness in dB of the mix audio (items, frames)
			['mic_loudness_arr'] : loudness in dB of the mic audio (items, samples)
        """
		dataset = {}
	
		print("Extracting track data...")
		for id in tqdm_notebook(track_ids):
			track_data = self.get_6_string_data_from_track(id, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			for key in track_data.keys():
				if key not in dataset:
					dataset[key] = [track_data[key]]
				else:
					dataset[key].append(track_data[key])

		print("Compiling dataset...")
		for key in dataset.keys():
			dataset[key] = np.concatenate(dataset[key])

		return dataset

	def create_midi_conditioning_dataset(self, track_ids, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a list of track IDs, sr, frame rate and item duration, compute audio and label arrays for a full midi conditioning dataset

        Parameters
        ----------
		track_ids : list of strings
        	ID # of tracks
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset

        dataset: dict of dicts
			['mix_audio'] : mix of the monophonic string audio (items, samples)
			['mic_audio'] : audio from the room mic (items, samples)
			['conditioning'] : midi-like conditioning (items, frames, num_string=6, pitch/velocity=2)
        """
		dataset = {}
	
		print("Extracting track data...")
		for id in tqdm_notebook(track_ids):
			track_data = self.get_audio_and_MIDI_conditioning_from_track(id, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			for key in track_data.keys():
				if key not in dataset:
					dataset[key] = [track_data[key]]
				else:
					dataset[key].append(track_data[key])

		print("Compiling dataset...")
		for key in dataset.keys():
			dataset[key] = np.concatenate(dataset[key])

		return dataset

	def save_dataset(self, dataset, name, save_path=DATASETS_PATH):
		"""
		Save a dataset (dict of numpy arrays) 

        Parameters
        ----------
		dataset : dictionary of numpy arrays
        	dataset to be saved
        name : string
			name of dataset
		save_path : string
			path of where to save. Will save as save_path/name

        """	
		file = os.path.join(save_path, name)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		np.savez(file, **dataset) # ** unpacks dictionary as separate keyword arguments

	def create_and_save_1_string_datasets_from_splits(self, splits, guitar_string_letters=GUITAR_STRING_LETTERS, save_path=DATASETS_PATH, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Save a dataset (dict of numpy arrays) 

        Parameters
        ----------
		splits : dictionary of (name : list of track ids) items
        	Dictionary containing track id lists for desired train/test split
		splits : tuple or list of strings (or just a string)
        	Tuple of guitar string names to make datasets for (or 1 single string)
		save_path : string
			Path of where to save datasets
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the datasets

        """	
		# if guitar_string_letters is not a list (or tuple), assume we have one letter and put it in a list
		if not(type(guitar_string_letters) is tuple or type(guitar_string_letters) is list):
			guitar_string_letters = [guitar_string_letters]

		for split_name, track_list in tqdm_notebook(splits.items()):
			for guitar_string in tqdm_notebook(guitar_string_letters):
				dataset = self.create_1_string_dataset(track_list, guitar_string, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
				save_name = split_name + "_" + guitar_string + "-string" + f"_{item_dur}s" + "_crepe" if self.use_crepe_f0_labels else ""
				self.save_dataset(dataset, save_name, save_path=save_path)
	
	def create_and_save_6_string_datasets_from_splits(self, splits, save_path=DATASETS_PATH, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Create and save a 6-string dataset (dict of numpy arrays) 

        Parameters
        ----------
		splits : dictionary of (name : list of track ids) items
        	Dictionary containing track id lists for desired train/test split
		save_path : string
			Path of where to save datasets
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the datasets

        """	
		for split_name, track_list in tqdm_notebook(splits.items()):
			dataset = self.create_6_string_dataset(track_list, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			save_name = split_name + "_" + "hex" + f"_{item_dur}s" + ("_crepe" if self.use_crepe_f0_labels else "")
			self.save_dataset(dataset, save_name, save_path=save_path)

	def create_and_save_midi_conditioning_datasets_from_splits(self, splits, save_path=DATASETS_PATH, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Create and save a midi conditioning dataset (dict of numpy arrays) 

        Parameters
        ----------
		splits : dictionary of (name : list of track ids) items
        	Dictionary containing track id lists for desired train/test split
		save_path : string
			Path of where to save datasets
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the datasets

        """	
		for split_name, track_list in tqdm_notebook(splits.items()):
			dataset = self.create_midi_conditioning_dataset(track_list, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			save_name = split_name + ("_crepe" if self.use_crepe_f0_labels else "") + "_" + "gset-midi" + f"_{item_dur}s"
			self.save_dataset(dataset, save_name, save_path=save_path)


	def get_6_string_data_from_track_unchunked(self, track_id, sr=SR, frame_rate=FRAME_RATE):
		"""
		Given a track ID and duration (and sampling and frame rates), compute a dict of arrays of unchunked audio, freqs, voicing, and loudness by string. This data is unchunked, lasting the full duration of the track rather than being segmented.

        Parameters
        ----------
		track_id : string
        	ID # of track
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations

        Returns
        ----------
		track_data: dict of dicts
			['E_audio'] : numpy array (samples)
				Cleaned isolated audios of E string from track
			['E_f0'] : numpy array (frames)
				F0 annotations of E string from track
				from GuitarSet labels unless self.use_crepe_f0-labels is true
			['E_voicing'] : numpy array (frames)
				Binary voicings of E string from track
			['E_loudness'] : numpy array (frames)
				Loudnesses of single string from track
			['A...'] (same keys as above)
			['D...'] (same keys as above)
			['G...'] (same keys as above)
			['B...'] (same keys as above)
			['e...'] (same keys as above)
			['mix_audio'] : mix of the monophonic string audio (samples)
			['mic_audio'] : audio from the room mic (samples)
			['mix_loudness'] : loudness in dB of the mix audio (frames)
			['mic_loudness'] : loudness in dB of the mic audio (samples)
        """

		# load track from ID
		track = self.tracks[track_id]

		# create empty dict for all string data
		track_data = {}
	
		# load mix, mic and hex audios
		mix_audio, mix_sr = track.audio_mix
		if mix_sr != sr:
			track_data['mix_audio'] = librosa.resample(mix_audio, orig_sr=mix_sr, target_sr=sr)
		mic_audio, mic_sr = track.audio_mic
		if mic_sr != sr:
			track_data['mic_audio'] = librosa.resample(mic_audio, orig_sr=mic_sr, target_sr=sr)

		track_dur = len(track_data['mix_audio']) / sr # in seconds
		track_frames = int(np.ceil(track_dur * frame_rate))

		# extract loudness expects frames first, so unsqueeze dim at 0 and squeeze it away afterward
		print("Extracting mix loudness...")
		track_data['mix_loudness'] = gset_proc.extract_loudness(np.expand_dims(mix_audio, 0), sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=track_frames).squeeze(0)
		print("Extracting mic loudness...")
		track_data['mic_loudness'] = gset_proc.extract_loudness(np.expand_dims(mic_audio, 0), sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=track_frames).squeeze(0)


		# load hex audio
		audio_hex, hex_sr = track.audio_hex_cln

		for string_let in GUITAR_STRING_LETTERS:
			
			string_audio = audio_hex[GUITAR_STRING_LETTERS.index(string_let)]
			if hex_sr != sr:
				string_audio = librosa.resample(string_audio, orig_sr=hex_sr, target_sr=sr)
			
			string_pitch_contour = track.pitch_contours[string_let]

			# if no pitch labels for the string, put all 0s
			if string_pitch_contour == None:
				f0 = np.zeros((track_frames,))
				voicing = np.zeros((track_frames,))

			else:
				# resample pitch contour to desired frame rate
				string_pitch_contour = gset_proc.resample_labels(string_pitch_contour, target_frame_rate=frame_rate)
				if self.use_crepe_f0_labels: # overwrite f0 label with crepe labels
					string_crepe_pitch_contour = gset_proc.crepe_compute_f0(string_audio, string_pitch_contour, sr=sr, frame_rate=frame_rate)

				# pad freqs and voicings with zero to be an appropriate length
				f0 = crop_or_pad(string_pitch_contour.frequencies, size=track_frames)
				voicing = crop_or_pad(string_pitch_contour.voicing, size=track_frames)

			print(f"Extracting {string_let}-string loudness...")
			loudness = gset_proc.extract_loudness(np.expand_dims(string_audio, 0), sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=track_frames).squeeze(0)
			
			track_data[f"{string_let}_audio"] = string_audio
			track_data[f"{string_let}_f0"] = f0
			track_data[f"{string_let}_voicing"] = voicing
			track_data[f"{string_let}_loudness"] = loudness

		print("Done.")
	
		return track_data

	def get_audio_and_MIDI_conditioning_from_track_unchunked(self, track_id, sr=SR, frame_rate=FRAME_RATE):
		"""
		Given a track ID, item duration (and sampling and frame rates), compute a conditioning array inspired by the work in:
		https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf. Conditioning contains information about note onsets and pitches.

		Conditioning is for the whole track, not 'chunked' into items of equal length.

        Parameters
        ----------
		track_id : string
        	ID # of track
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
        Returns
        ----------
		track_data : dict
        	track_data['conditioning'] : np array (frames, num_strings=6, pitch_or_onset=2)
				midi pitch and onset velocity for each string
			track_data['mic_audio'] : np array (samples,)
				audio from room microphone
			track_data['mix_audio'] : np array (samples,)
				audio mixture summed from hexaphonic signal		
        """

		# load track from ID
		track = self.tracks[track_id]

		# create empty dict for all string data
		track_data = {}

		# load mix, mic and hex audios
		mix_audio, mix_sr = track.audio_mix
		if mix_sr != sr:
			mix_audio = librosa.resample(mix_audio, orig_sr=mix_sr, target_sr=sr)
		mic_audio, mic_sr = track.audio_mic
		if mic_sr != sr:
			mic_audio = librosa.resample(mic_audio, orig_sr=mic_sr, target_sr=sr)

		track_seconds = len(mix_audio) / sr # seconds
		track_frames = int(np.ceil(track_seconds * frame_rate)) # frames

		# create empty conditioning array 
		conditioning = np.zeros((track_frames, 6, 2)) # (frames, strings, pitch/onset_velocity)	

		# load hex audio
		audio_hex, hex_sr = track.audio_hex_cln

		for string_idx, string_let in enumerate(GUITAR_STRING_LETTERS):

			# load the individual string audio, resampling if necessary
			string_audio = audio_hex[string_idx]
			if hex_sr != sr:
				string_audio = librosa.resample(string_audio, orig_sr=hex_sr, target_sr=sr)

			### compute the midi pitch array
					
			# get the string's pitch contour
			string_pitch_contour = track.pitch_contours[string_let]		

			# if no pitch labels for the string, move onto the next string
			if string_pitch_contour == None:
				continue

			# resample pitch contour to desired frame rate
			string_pitch_contour = gset_proc.resample_labels(string_pitch_contour, target_frame_rate=frame_rate)

			# optionally use crepe to compete f0 labels
			if self.use_crepe_f0_labels:
				string_pitch_contour = gset_proc.crepe_compute_f0(string_audio, string_pitch_contour, sr=sr, frame_rate=frame_rate)

			# pad freqs with zero to be an appropriate length
			freqs = crop_or_pad(string_pitch_contour.frequencies, size=track_frames)
			# convert from hz to midi, clip between 0 and 127
			midi_pitches = np.clip(librosa.hz_to_midi(freqs), 0, 127) # (frames,)



			# compute framewise loudness of the string audio, expand to get batch dimension
			# make sure to set scale to MIDI-range to true to get loudness in [0, 127]
			string_loudness = gset_proc.extract_loudness(np.expand_dims(string_audio, axis=0), 
														sr=sr, 
														hop_length=int(round(sr/frame_rate)), 
														desired_len=track_frames, 
														scale_to_MIDI_range=True)[0]		
			
			# initialize stringwise onset velocity 
			onset_vel = np.zeros((track_frames)) # shape: (frames,)

			# get just the onset times of all the notes on the given string
			onset_times = track.notes[string_let].intervals[:,0] 
			# convert time to frames
			onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length=int(round(sr/frame_rate)))

			# get the onset velocities from the string loudness
			onset_vel[onset_frames] = string_loudness[onset_frames]

			
			# store conditioning
			conditioning[:,string_idx,0] = midi_pitches
			conditioning[:,string_idx,1] = onset_vel

		
		# store track data in dict
		track_data['conditioning'] = conditioning # shape : (frames, num_strings=6, pitch_or_onset=2)
		track_data['mic_audio'] = mic_audio # shape : (samples,)
		track_data['mix_audio'] = mix_audio # shape : (samples,)
		
		return track_data