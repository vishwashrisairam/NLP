import os
import io
import sys

class Tagger:

	def __init__(self):
		self.initial_tag_probability = None
		self.transition_probability = None
		self.emission_probability = None

		self.sentence=None
		self.tag_dictionary={}
		self.words_dictionary={}

	def load_corpus(self, path):
		if not os.path.isdir(path):
			sys.exit("Input path is not a directory")
		output=[]
		for filename in os.listdir(path):
			filename = os.path.join(path, filename)
			try:
				reader = io.open(filename)

				"""
				YOUR CODE GOES HERE: Complete the rest of the method so that it outputs a list of lists as described in the question
				"""
				lines=reader.read().splitlines()
				lines=filter(lambda x: x!='',lines)
				lines=[i.split(' ') for i in lines]
				lines=[[tuple(i.split('/')) for i in l[:-1] ] for l in lines]
				output.extend(lines)

			except IOError:
				sys.exit("Cannot read file")
			
			
		return output

	def initialize_probabilities(self, corpus):
		if type(corpus) != list:
			sys.exit("Incorrect input to method")
		"""
		YOUR CODE GOES HERE: Complete the rest of the method so that it computes the probability matrices as described in the question
		"""
		#Calculation of initial probability 
		tag_keys=self.tag_dictionary.keys()
		num_tags=len(tag_keys)
		tot_count_tags=0
		for key in self.tag_dictionary.keys():
			tot_count_tags+=self.tag_dictionary[key]

		#Calculate the count at the start of the sentence 
		tag_dictionary_sos={}
		for sentences in corpus:
			tup_word,tup_tag=sentences[0]
			if tup_tag in tag_dictionary_sos:
				cnt=tag_dictionary_sos[tup_tag]
				tag_dictionary_sos[tup_tag]=cnt+1
			else:
				tag_dictionary_sos[tup_tag]=1

		self.initial_tag_probability={}
		for key in tag_keys:
			self.initial_tag_probability[key]=(tag_dictionary_sos[key]/tot_count_tags)
		# print(self.initial_tag_probability)
		
		#Calculation of tag transition probability

		#Initialization
		self.transition_probability={}
		for i in tag_keys:
			self.transition_probability[i]={}
			for j in tag_keys:
				self.transition_probability[i][j]=0
		
		#Count of the tag pairs 
		prev_tag=''
		for sentences in corpus:
			for i,tup in enumerate(sentences):
				if len(tup)==2:  
					if i==0 and prev_tag!='':
						cnt=self.transition_probability[prev_tag][tup[1]]
						self.transition_probability[prev_tag][tup[1]]=cnt+1
					if i+1 <len(sentences):
						next_tup=sentences[i+1]
						cnt=self.transition_probability[tup[1]][next_tup[1]]
						self.transition_probability[tup[1]][next_tup[1]]=cnt+1
					if i== len(sentences)-1:
						prev_tag=tup[1]
		
		#Dividide it with count of each tag to get the probability
		for key in self.transition_probability.keys():
			key_dict=self.transition_probability[key]
			curr_key_count=self.tag_dictionary[key]
			
			for j in key_dict.keys():
				self.transition_probability[key][j]=(self.transition_probability[key][j]+1)/(curr_key_count + num_tags) #Add smoothing

		# print(self.transition_probability)

		#Calcualte the emission probabilities 

		#Initialization
		word_set=list(set(self.sentence.split()))
		self.emission_probability={}
		for i in self.tag_dictionary.keys():
			self.emission_probability[i]={}
			for j in word_set:
				self.emission_probability[i][j]=0
		
		#Take the count of all the possible words for a given tag from the corpus
		"""
		for sentences in corpus:
			for tup in sentences:
				if len(tup)==2:
					word,tag=tup
					if word in self.emission_probability[tag]:
						wrd_cnt=self.emission_probability[tag][word]
						self.emission_probability[tag][word]=wrd_cnt+1
					else:
						self.emission_probability[tag][word]=1
		"""
		for tag in self.tag_dictionary.keys():
			for word in word_set:
				self.emission_probability[tag][word]=self.get_word_count_corpus(word,tag,corpus)

		#Divide it with count of each tag to get the probability
		for key in self.emission_probability.keys():
			key_dict=self.emission_probability[key]
			curr_key_count=self.tag_dictionary[key]

			for j in key_dict.keys():
				self.emission_probability[key][j]=(self.emission_probability[key][j]+1)/(curr_key_count + num_tags) #Add smoothing

		# print(self.emission_probability)


	def viterbi_decode(self, sentence):
		if type(sentence) != str:
			sys.exit("Incorrect input to method")
		"""
		YOUR CODE GOES HERE: Complete the rest of the method so that it computes the most likely sequence of tags as described in the question
		"""
		words=sentence.split()
		M=len(words)
		tag_keys=self.tag_dictionary.keys()
		num_tags=len(tag_keys)
		
		viterbi=[[0 for i in range(len(words)) ] for j in range(num_tags)]
		b_ptr=[[0 for i in range(len(words)) ] for j in range(num_tags)]

		for j,tag in enumerate(tag_keys): #base case
			viterbi[j][0] = self.initial_tag_probability[tag]*self.emission_probability[tag][words[0]]
		

		for t in range(1,M):
			for j,tag_j in enumerate(tag_keys):
				max_val,max_i=viterbi[0][t-1]*self.transition_probability[next(iter(self.tag_dictionary))][tag_j]*self.emission_probability[tag_j][words[t]],0
				for i,tag_i in enumerate(tag_keys):
					prob_val=viterbi[i][t-1]*self.transition_probability[tag_i][tag_j]*self.emission_probability[tag_j][words[t]]
					if max_val < prob_val:
						max_val,max_i=prob_val,i
				viterbi[j][t]=max_val
				b_ptr[j][t]=max_i
		
		#Final state probability
		omx,omi=viterbi[0][M-1],0
		for i in range(1,num_tags):
			if viterbi[i][M-1] >omx:
				omx,omi=viterbi[i][M-1],i
		# print(viterbi)
		
		#Backtrace
		i,p=omi,[omi]
		for j in range(M-1,-1,-1):
			i=b_ptr[i][j]
			for j,tag in enumerate(tag_keys):
				if j==i:
					p.append(tag)
		p=p[1:]
		return p

	def initialize_counts(self,corpus):
		"""  This method will set the counts for each word and tag in the corpus"""
		for sentences in corpus:
			for tup in sentences:
				if len(tup)==2:
					word,tag=tup
					if word in self.words_dictionary:
						cnt=self.words_dictionary[word]
						self.words_dictionary[word]=cnt+1
					else:
						self.words_dictionary[word]=1
					
					if tag in self.tag_dictionary:
						cnt=self.tag_dictionary[tag]
						self.tag_dictionary[tag]=cnt+1
					else:
						self.tag_dictionary[tag]=1
		# print(self.words_dictionary)
		# print(self.tag_dictionary)

	def get_word_count_corpus(self,word,tag,corpus):
		count=0
		for sentences in corpus:
			for tup in sentences:
				if tup==(word,tag):
					count+=1
		return count

		