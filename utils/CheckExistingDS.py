train_entitites = []
train_relations = []
train_lines = []
def readTrain(type):
    f1=open('fb15k/'+type+'/train.txt', 'r')
    train_lines.clear()
    train_relations.clear()
    train_entitites.clear()
    all_lines = f1.readlines()
    train_lines.extend(all_lines)
    #print(train_lines[0:4])
    all_lines = [line.split() for line in all_lines]
    #print(train_lines[0:4])
    for line in all_lines:
        if line[0] not in train_entitites:
            train_entitites.append(line[0])
        if line[2].rstrip() not in train_entitites:
            train_entitites.append(line[2].rstrip())
        if line[1] not in train_relations:
            train_relations.append(line[1])
    print('train_lines', len(train_lines))
    print('train_relations', len(train_relations))
    print('train_entities', len(train_entitites))

def WriteTransductive(type, mode):
    #b_e = 0
    #print(train_lines[0:4])
    output_lines = []
    f1 = open('fb15k/'+type+'/'+mode+'.txt', 'r')
    #f2 = open('fb15k/fixedDS/'+type+'/'+mode+'.txt', 'w')
    all_lines = f1.readlines()
    print('total '+type +'_'+ mode + ' lines ' +str(len(all_lines)))
    all_lines = [line.split() for line in all_lines]
    for line in all_lines:
        #entities exiting in train_entities
        #relation exsiting in train_relations
        tempL = line[0] + '\t' + line[1] + '\t' + line[2] + '\n'
        if (line[0] in train_entitites and line[2].rstrip() in train_entitites and line[1] in train_relations):
            #the whole tuple should not appear in train 
            if tempL not in train_lines:
                #f2.write(tempL)
                output_lines.append(tempL)
            else:
                print(tempL)
        #else:
            #b_e = b_e + 1
    print('extracted ' + type + '_'+mode +' lines '+ str(len(output_lines)))
    #print('b_e', b_e)

def WriteInductive(type, mode):
    #b_e = 0
    #print(train_lines[0:4])
    output_lines = []
    f1 = open('fb15k/'+type+'/'+mode+'.txt', 'r')
    f2 = open('fb15k/fixedDS/Inductive/'+type+'/'+mode+'.txt', 'w')
    all_lines = f1.readlines()
    print('total '+type +'_'+ mode + ' lines ' +str(len(all_lines)))
    all_lines = [line.split() for line in all_lines]
    for line in all_lines:
        #entities exiting in train_entities
        #relation exsiting in train_relations
        tempL = line[0] + '\t' + line[1] + '\t' + line[2] + '\n'
        if (line[0] not in train_entitites and line[2].rstrip() not in train_entitites and line[1] in train_relations):
            #the whole tuple should not appear in train 
            if tempL not in train_lines:
                f2.write(tempL)
                output_lines.append(tempL)
            else:
                print(tempL)
        #else:
            #b_e = b_e + 1
    print('extracted ' + type + '_'+mode +' lines '+ str(len(output_lines)))


types = ['Symmetry/People', 'inverse', 'AntiSymmetry', 'Inference']
for i in types:
    print(i)
    readTrain(i)
    #WriteTransductive(i, 'valid')
    #WriteTransductive(i, 'test')
    WriteInductive(i, 'valid')
    WriteInductive(i, 'test')