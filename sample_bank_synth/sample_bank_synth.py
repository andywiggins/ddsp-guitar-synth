# naive synthesis approach

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.fft as fft
import librosa
from globals import HOP_LENGTH

class InstrumentSourceSet(object):
  """Class to store inst source info"""

  def __init__(self, arr_load_path = "/content/drive/MyDrive/Research/Projects/GuitaristNet/Data/single_notes_gset.npy"):
    self.note_arr = torch.from_numpy(np.load(arr_load_path), ) # (# guitars, # notes, t)
    self.n_guitars, self.n_notes, self.n_samples = self.note_arr.shape
    self.note_names = ["E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", 
              "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4",
              "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5",
              "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"]
    if len(self.note_names) != self.n_notes:
      print("error: global note name list does not match len of single note array")

  def get_note(self, note_name, guitar_num):
    note_index = self.note_names.index(note_name)
    return self.note_arr[guitar_num][note_index]

  def __str__(self):
        return ('n_notes: %d, n_guitars:' % self.n_notes) + str(self.n_guitars)


class ZeroInserter(nn.Module):
    """Insert zeros for upsampling the transcription"""

    def __init__(self, insertion_rate):
        super(ZeroInserter, self).__init__()
        self.insertion_rate = insertion_rate

    def forward(self, downsampled_y):
        batch, ch, time = downsampled_y.shape
        upsampled_y = []
        for ch_idx in range(ch):
            ds_y = downsampled_y[:, ch_idx:ch_idx + 1, :]  # (batch, 1, time)
            us_y = torch.cat((ds_y,
                              torch.zeros((batch, self.insertion_rate - 1, time),
                                          device=downsampled_y.device)),
                             dim=1)  # (batch, insert_rate, time)
            us_y = us_y.transpose(2, 1)  # (b, t, insert_rate)
            us_y = torch.reshape(us_y, (batch, 1, self.insertion_rate * time))  # (b, 1, t*insert_rate)
            upsampled_y.append(us_y)

        upsampled_y = torch.cat(upsampled_y, dim=1)
        return upsampled_y
    
def fft_convolve(x, h):
  # x and h are 1D tensors to be convolved
  # convolution is convolved along the -1 dimension
  # so, signal can be: (batch, ch=1, time)
  # and kernel: (time)
  
  x=x.type(torch.float32)
  h=h.type(torch.float32)

  nx = x.shape[-1]
  nh = x.shape[-1]

  length = nx + nh - 1                      # min length is M + N - 1
  nfft = 2**(length - 1).bit_length()       # get next power of 2
  
  X = fft.fft(x, n=nfft)
  H = fft.fft(h, n=nfft)

  Y = H * X
  y = fft.ifft(Y).real
  
  return y

class DecayEnvelopeGenerator(nn.Module):
    """takes in tensor of decay params, returns tensor of envelopes
    input: (frames)
    output: (time) 
    """

    def getDecayEnvelopeBatch(self, decayParamsBatch):
        ITEM_SAMPLES = 32000
        ''' decayParams: 2D tensor of decay params per batch item, which indicate note length and occur at onset location.(batch, # frames)'''
        dev = decayParamsBatch.device
        envs = torch.zeros(decayParamsBatch.shape[1], decayParamsBatch.shape[0], ITEM_SAMPLES * 2)
        for frameInd in range(decayParamsBatch.shape[1]):
            decayParamBatch = decayParamsBatch[:,frameInd] # a single column of decay params at the given frame
            ones = torch.ones((decayParamsBatch.shape[0], ITEM_SAMPLES), device=dev) # shape of batch of envs, to multiply by arange
            timeTensor = torch.arange(0, 1, 1.0/ITEM_SAMPLES, device=dev) * ones
            alphas = decayParamBatch * -100 # map [0,1] onto [0,-100]
            exps = torch.exp(alphas[:,None] * timeTensor) # none adds a dimension of 1 which allows broadcasting
            norm_exps = F.normalize(exps * decayParamBatch[:,None],p=float("inf"),dim=1) # multiply by param, then normalize, to get zeros squashed
            startSample = librosa.frames_to_samples(frameInd, hop_length=512)
            envs[frameInd][:, startSample:startSample+ITEM_SAMPLES] = norm_exps
        max_over_frames = torch.max(envs, dim=0).values
        return max_over_frames[:,:ITEM_SAMPLES]


    def forward(self, decayParamsTrans):
        """decayParams: (batch, pitches, frames)
        return: (batch, pitches, time)"""
        ITEM_SAMPLES = 32000
        ret = torch.zeros((decayParamsTrans.shape[0], decayParamsTrans.shape[1], ITEM_SAMPLES), device=decayParamsTrans.device)

        for pitch in range(decayParamsTrans.shape[1]):
            decayParams = decayParamsTrans[:,pitch,:]
            ret[:,pitch] = self.getDecayEnvelopeBatch(decayParams)

        return ret
    
    
#changed name to not include "drum" -AFW
class FastSynthesizer(nn.Module):
    """Freq-domain convolution-based synthesizer"""

    def __init__(self, inst_srcset, guitar_index=0):
        super(FastSynthesizer, self).__init__()
        self.inst_srcset = inst_srcset # Instrument Source Set class
        self.n_notes = inst_srcset.n_notes
        self.guitar_index = guitar_index
        
        self.note_names = ["E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", 
              "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4",
              "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5",
              "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"]

    def forward(self, midis):
        """
        midis: (batch, inst, time)
        returned tracks: (batch, inst, time)
        """
        device_ = midis[0].device
        
        guitar_index = self.guitar_index
        
        
        rv_insts = [self.inst_srcset.get_note(note_name, guitar_index).to(device_) for note_name in self.note_names]

        igw = [1. / rv_i.abs().sum() for rv_i in rv_insts]
        igw = torch.tensor(igw, device=device_) / max(igw)
        
        tracks = []
        for i in range(self.n_notes):
            md = midis[:, i:i + 1, :]
            # track = fast_conv1d(md, rv_insts[i].expand(1, 1, -1))
            track = fft_convolve(md, rv_insts[i].expand(1, 1, -1)) # AFW 2/1/22
            tracks.append(track)
               
        # return torch.cat(tracks, dim=1)
        return torch.cat(tracks, dim=1)[:,:,:midis.shape[2]] # trim shape to match midis AFW 2/2/22

class FastEnvSynthesizer(nn.Module):
    """Freq-domain convolution-based synthesizer"""

    def __init__(self, inst_srcset):
        super(FastEnvSynthesizer, self).__init__()
        self.inst_srcset = inst_srcset # Instrument Source Set class
        self.n_notes = inst_srcset.n_notes
        
        self.note_names = ["E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3", 
              "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4",
              "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5",
              "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"]

    def forward(self, midis, envs):
        """
        midis: (batch, inst, time)
        envs: (batch, inst, time)
        returned tracks: (batch, inst, time)
        """
        ITEM_SAMPLES = 32000

        device_ = midis[0].device
        
        guitar_index = 0
        
        
        rv_insts = [self.inst_srcset.get_note(note_name, guitar_index).to(device_) for note_name in self.note_names]

        igw = [1. / rv_i.abs().sum() for rv_i in rv_insts]
        igw = torch.tensor(igw, device=device_) / max(igw)
        
        tracks = []
        for i in range(self.n_notes):
            md = midis[:, i:i + 1, :]
            env = envs[:, i:i + 1, :]
            track = fft_convolve(md, rv_insts[i].expand(1, 1, -1))
            track = track[:,:,:ITEM_SAMPLES] # crop to length ITEM_SAMPLES
            enveloped_track = track * env
            tracks.append(enveloped_track)
               
        # return torch.cat(tracks, dim=1)
        return torch.cat(tracks, dim=1)[:,:,:midis.shape[2]] # trim shape to match midis AFW 2/2/22
        
class Mixer(nn.Module):
    """Sum mixer"""

    def forward(self, tracks, group_by=None):   ## AFW: I could use group_by to potentially lump pitch classes together
        """tracks: (batch, inst, time)
        return: (batch, time)"""
        if group_by:
            return tracks[:, group_by, :].sum(dim=1)
        else:
            return tracks.sum(dim=1)


class naiveSynth(nn.Module):
    def __init__(self, guitar_srcset = InstrumentSourceSet(), guitar_index=0):
        """
        inst_srcs: 2d torch tensor, instrument waveforms to use in the resynthesis
        """
        super(naiveSynth, self).__init__()

        self.guitar_srcset = guitar_srcset
        self.n_notes = self.guitar_srcset.n_notes  # 44

        self.zero_inserter = ZeroInserter(insertion_rate=HOP_LENGTH) # hop length? 12/2/21
        self.synthesizer = FastSynthesizer(self.guitar_srcset, guitar_index=guitar_index)
        self.mixer = Mixer()

    def forward(self, sparse_tr):
        """x: (batch, num_notes, frames)
        """
        upsampled_tr = self.zero_inserter(sparse_tr)
        note_signals = self.synthesizer(upsampled_tr)
        x_hat = self.mixer(note_signals)
        
        return x_hat

class naiveDecaySynth(nn.Module):
    def __init__(self, guitar_srcset = InstrumentSourceSet()):
        """
        inst_srcs: 2d torch tensor, instrument waveforms to use in the resynthesis
        """
        super(naiveDecaySynth, self).__init__()

        self.guitar_srcset = guitar_srcset
        self.n_notes = self.guitar_srcset.n_notes  # 44

        self.decay_envelope_generator = DecayEnvelopeGenerator()
        self.zero_inserter = ZeroInserter(insertion_rate=HOP_LENGTH) # hop length: default 512
        self.synthesizer = FastEnvSynthesizer(self.guitar_srcset)
        self.mixer = Mixer()

    def forward(self, x, decayParams):
        """x: (batch, num_notes, frames)
          decayParams: (batch, num_notes, frames) 
        """
        upsampled_tr = self.zero_inserter(x)
        envs = self.decay_envelope_generator(decayParams)
        note_signals = self.synthesizer(upsampled_tr, envs)
        x_hat = self.mixer(note_signals)
        
        return x_hat


class naiveOnsetFrameSynth(nn.Module):
    '''A synthesizer that produces audio given an onset prediction and a frame prediction'''
    def __init__(self, guitar_srcset = InstrumentSourceSet(), guitar_index=0):
        """
        inst_srcs: 2d torch tensor, instrument waveforms to use in the resynthesis
        """
        super(naiveOnsetFrameSynth, self).__init__()

        self.guitar_srcset = guitar_srcset
        self.n_notes = self.guitar_srcset.n_notes  # 44

        self.zero_inserter = ZeroInserter(insertion_rate=HOP_LENGTH) # hop length
        self.synthesizer = FastSynthesizer(self.guitar_srcset, guitar_index=guitar_index)
        self.frame_upsampler = nn.Upsample(scale_factor=HOP_LENGTH, mode='linear') # 512 is default hop length
        self.mixer = Mixer()

    def forward(self, onsets_tr, frames_tr, ignore_envs=False):
        """onsets_tr: (batch, num_notes, frames)
          frames_tr: (batch, num_notes, frames) 
        """
        upsampled_tr = self.zero_inserter(onsets_tr)
        note_signals = self.synthesizer(upsampled_tr)
        envs = self.frame_upsampler(frames_tr) 
        if ignore_envs:
            envs = 1
        x_hat = self.mixer(note_signals * envs)
        
        return x_hat