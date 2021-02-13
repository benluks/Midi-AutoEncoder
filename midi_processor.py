"""
midi_processor.py
~~~~~~~~~~~~~~~~

For all your midi processing needs!

"""

from mido import MidiFile, Message, MidiTrack, MetaMessage

from os import listdir
from os.path import  abspath, expanduser as eu, join

import numpy as np
import matplotlib.pyplot as plt

HOME = eu('~')
PATH = join(HOME, 'VGmidi')

files = listdir(PATH)

all_midi_matrices = []



### MidiMatrix class
###
### This is a class that extends the `mido.MidiFile` module. It's purpose is to convert 
### .mid files into the numpy arrays that are needed as input for our network

class MidiMatrix(MidiFile):

	def __init__(self, file_index):
		super().__init__(abspath(join(PATH, files[file_index])))
		
		self.time_sig_msg = next((msg for msg in self.tracks[0] if msg.type == 'time_signature'), None)
		self.numerator, self.denominator = (self.time_sig_msg.numerator, self.time_sig_msg.denominator) if self.time_sig_msg and self.time_sig_msg.numerator != 1 else (4, 4)
		self.time_signature = (self.numerator, self.denominator)

		# the measure coefficient helps determine how many tempo or clicks there are per measure.
		# This is important for our purposes. It's (numerator * 4) / denominator. This, when multiplied 
		# by tempo (in microseconds per beat) or ticks_per_beat converts per_beat to per_measure.
		self.measure_coeff = int(4 * self.numerator / self.denominator)
		
		# note: the 'beat' in `ticks_per_beat` refers to the -- quarter note --
		self.ticks_per_measure = int((self.numerator * self.ticks_per_beat) / (self.denominator / 4))
		self.ticks_per_step = (self.ticks_per_measure / 96)



	def _secs_to_steps(self, tempo, cum_time):
		"""
		Take a time and convert it to an index in our matrix.
		Note the tempo is in microseconds per beat and cum_time in 
		seconds.
		"""
		microseconds_per_measure = self.measure_coeff * tempo
		seconds_per_measure = microseconds_per_measure / 1000000
		secs_per_step = seconds_per_measure / 96

		step_idx_float = cum_time / secs_per_step

		# round to the nearest step
		step_idx = int(step_idx_float) + int((step_idx_float - int(step_idx_float)) // 0.5)
		return step_idx


	def _fill_by_ticks(self):
		"""
		If no tempo information is given, then the midi matrix has to be filled track-by-track 
		with reference to ticks (explained here: 
		https://mido.readthedocs.io/en/latest/midi_files.html#about-the-time-attribute). This is 
		computationally more expensive and therefore disfavourable.
		"""
		
		matrix = np.zeros((16, 96, 96))

		for track in self.tracks[1:]:
			
			cum_time_in_ticks = 0	# you have to reset the time at the beginning of each track
			
			for message in track:
				cum_time_in_ticks += message.time
				
				if message.type == 'note_on' and message.channel != 9 and message.velocity != 0:

					pitch_idx = 116 - message.note
					num_steps_float = cum_time_in_ticks / self.ticks_per_step
					step_idx = int(num_steps_float) + int((num_steps_float - int(num_steps_float)) // 0.5)

					measure_idx = step_idx // 96 

					if measure_idx > 15:
						break

					matrix[measure_idx, pitch_idx, step_idx % 96] = message.velocity

		return matrix




	def mid_to_matrix(self):
		"""
		Converts a .mid file into a np arrays of shape ((max) 16, 96, 96) for each
		block of 16 measures (first dimension is less than 16 if fewer measures remain).
		The lowest note is A0, value = 21, highest is G#8, value = 116. The value is 
		computed by taking message.value - 21.
		"""

		tempo = next((msg.tempo for msg in self if msg.type == 'set_tempo'), None)	# tempo in microseconds per beat
		
		if not tempo:
			# If no available tempo, then use ticks to map notes
			matrix = self._fill_by_ticks()
			all_midi_matrices.append(matrix)
			return matrix


		cumulative_time = 0		# accumulate time in seconds
		# we can compute num steps by dividing cumulative_time_in_ticks by ticks_per_step.
		# we then round that number to the nearest integer and voila
		
		# allot zero matrix to be filled in
		matrix = np.zeros((16, 96, 96))
		# iterate over all the tracks. First track is only metadata -- not important
		for message in self:

			if message.type == 'end_of_track':
				break

			time_in_secs = message.time
			cumulative_time += time_in_secs
			
			if message.type == 'note_on' and message.channel != 9 or message.type == 'set_tempo':
				
				if message.type == 'set_tempo':
					tempo = message.tempo
				
				elif message.type == 'note_on' and message.velocity != 0:

					pitch_idx = 116 - message.note
					while pitch_idx > 95:
						pitch_idx -= 12
					while pitch_idx < 0:
						pitch_idx += 12
					
					step_idx = self._secs_to_steps(tempo, cumulative_time)
					# index of measure
					measure_idx = step_idx // 96
					# index of 16-measure sample
					sample_idx = measure_idx // 16
					
					if measure_idx > 15:	# finish after 16 measures
						break
				
					matrix[measure_idx, pitch_idx, step_idx % 96] = message.velocity
			

		all_midi_matrices.append(matrix)
		return matrix


# mid = MidiMatrix(-4)
# midmat = mid.mid_to_matrix()
# print(midmat[0, :, 6])
# m, p, s = np.where(midmat != 0)

# print(outfile.shape)

# axes = []
# fig=plt.figure()

# for i in range(8):
# 	plt.figure()
# 	plt.imshow(midmat[i])
# 	plt.title("Measure" + str(i+1))

# plt.show()


def matrix_to_mid(matrix, outfile_name):

	mid = MidiFile()

	# meta_track = MidiTrack()
	# mid.tracks.append(meta_track)
	# meta_track.append(MetaMessage('time_signature', numerator = 4, denominator = 4, time=0))
	# meta_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
	
	
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = 24

	cumulative_step = 0

	for m in range(16):
		for s in range(96):

			current_step = s + (96 * m)
			notes_at_step = matrix[m, :, s]
			notes = [(i, v) for (i, v) in enumerate(notes_at_step) if v != 0]

			for note, velocity in notes:
				# note that delta_time won't change after the first note is added 
				# which is what we're looking for. All notes are sounded at the same 
				# time, so we want all but the first note of a step to 
				delta_time = current_step - cumulative_step
				# make a new attack at specified notes
				track.append(Message('note_on', channel = 0, note = 116 - note, velocity = int(velocity), time = delta_time))

				cumulative_step += delta_time

			for note, velocity in notes:
				#release all of the notes
				track.append(Message('note_on', channel = 0, note = 116 - note, velocity = 0, time = 0))


	# meta_track.append(MetaMessage('end_of_track'))

	mid.save(outfile_name)


for i in range(50):
	mid = MidiMatrix(i)
	midmat = mid.mid_to_matrix()
	name = mid.filename.split('/')[-1]
	matrix_to_mid(midmat, "mmtests/" + name)
	print("%d completed, %s was successfully converted" % (i+1, name))




# plt.imshow(midmat[0])
# plt.show()







