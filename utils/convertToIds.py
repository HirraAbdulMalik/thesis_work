dir = '../fb15k/Symmetry/People/'
lines = []
unique_entities = []
unique_relations = []

def extract_entities_relations(fileName):
	k = 0
	with open(fileName, "r")as a_file:
		for line in a_file:
			sep_line = line.split()
			if sep_line[0] is not None and sep_line[0] not in unique_entities:
				unique_entities.append(sep_line[0])
			sep_line[2] = sep_line[2].rsplit()[0]
			if sep_line[2] is not None and sep_line[2] not in unique_entities:
				unique_entities.append(sep_line[2])
			if sep_line[1] is not None and sep_line[1] not in unique_relations:
				unique_relations.append(sep_line[1])

dict_entities = {}
dict_relations = {}
def write_Ids():
	i = 0
	print('writing entity to Ids')
	with open(dir + "/entity2Id.txt",'w')as a_file:
		#a_file.write(str(len(unique_entities)) + '\n')
		for line in unique_entities:
			#a_file.write(line + '\t' +str(i) + '\n')
			dict_entities[line] = i
			i = i + 1

	print('writing relation to Ids')
	i = 0
	with open(dir + "relation2Id.txt",'w')as a_file:
		#a_file.write(str(len(unique_relations)) + '\n')
		for line in unique_relations:
			#a_file.write(line + '\t' + str(i) + '\n')
			dict_relations[line] = i
			i = i + 1
def write_dicts():
	i = 0
	print('writing entity to dict_Ids')
	with open(dir + "/entities.dict", "w") as d_file:
		for line in unique_entities:
			d_file.write(str(i) + '\t' + line + '\n')
			dict_entities[line] = i
			i = i + 1

	print('writing relation to dict_Ids')
	i = 0
	with open(dir + "/relations.dict", "w") as d_file:
		for line in unique_relations:
			d_file.write(str(i) + '\t' + line + '\n')
			dict_relations[line] = i
			i = i + 1

def resolve_Ids(fileName):
	fileName = dir + fileName
	k = 0
	print('writing file to Id : ' + fileName)
	new_lines = []
	with open(fileName, "r")as a_file:
		for line in a_file:
			new_line = ''
			sep_line = line.split()
			new_line = str(dict_entities[sep_line[0]]) + '\t' + str(dict_entities[sep_line[2].rsplit()[0]]) + '\t' + str(dict_relations[sep_line[1]])
			new_lines.append(new_line + '\n')
			k = k + 1
	with open(fileName + '2id.txt', 'w') as new_file:
		new_file.write(str(k)+'\n')
		for line in new_lines:
			new_file.write(line)


#print('extracting from train file')
extract_entities_relations(dir + 'train.txt')
print('total unique_entities' + str(len(unique_entities)))
print('total unique_relations' + str(len(unique_relations)))
print('extracting from test file')
extract_entities_relations(dir + 'test.txt')
print('total unique_entities' + str(len(unique_entities)))
print('total unique_relations' + str(len(unique_relations)))
print('extracting from valid file')
extract_entities_relations(dir + 'valid.txt')
print('total unique_entities' + str(len(unique_entities)))
print('total unique_relations' + str(len(unique_relations)))
#write_dicts()
write_Ids()
#resolve_Ids('train.txt')
#resolve_Ids('test.txt')
#resolve_Ids('valid.txt')
resolve_Ids('negative_test_samples.txt')
