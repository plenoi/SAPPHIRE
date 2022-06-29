import numpy as np
import pandas as pd
from collections import Counter
import re
def read_fasta(file):
    line1 = open(file).read().split('>')[1:]
    line2= [item.split('\n')[0:-1] for item in line1] 
    fasta = [[item[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', ''.join(item[1:]).upper())] for item in line2]
    return fasta

def rearrange(positive, positive1):
    idx = []
    for i in range(len(positive)):
        idx.append(positive1[0].tolist().index(positive[i][0]))
    positive1s = positive1.values[idx,1:]
    return positive1s

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    #encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1 - gap):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def APAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = []
    records.append("#	A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V")
    records.append("Hydrophobicity	0.62	-2.53	-0.78	-0.9	0.29	-0.85	-0.74	0.48	-0.4	1.38	1.06	-1.5	0.64	1.19	0.12	-0.18	-0.05	0.81	0.26	1.08")
    records.append("Hydrophilicity	-0.5	3	0.2	3	-1	0.2	3	0	-0.5	-1.8	-1.8	3	-1.3	-2.5	0	0.3	-0.4	-3.4	-2.3	-1.5")
    records.append("SideChainMass	15	101	58	59	47	72	73	1	82	57	57	73	75	91	42	31	45	130	107	43")

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = np.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))
    
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        theta = []

        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                  range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]

        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Count(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def CTDC(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for g in range(1, len(groups) + 1):
            header.append(p + '.G' + str(g))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            c1 = Count(group1[p], sequence) / len(sequence)
            c2 = Count(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Count2(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, np.floor(0.25 * number), np.floor(0.50 * number), np.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDD(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')


    encodings = []
    header = []
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append(p + '.' + g + '.residue' + d)

    for i in fastas:
        name, sequence  = i[0], re.sub('-', '', i[1])
        code = []
        for p in property:
            code = code + Count2(group1[p], sequence) + Count2(group2[p], sequence) + Count2(group3[p], sequence)
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

file = open('PLS.py','w')
file.write('import numpy as np'+"\n")
file.write('from sklearn.cross_decomposition import PLSRegression'+"\n")
file.write('from sklearn.base import BaseEstimator, ClassifierMixin'+"\n")
file.write('class PLS(BaseEstimator, ClassifierMixin):'+"\n")
file.write('    def __init__(self):'+"\n")
file.write('        self.clf = PLSRegression(n_components=2)'+"\n")
file.write('    def fit(self, X, y):'+"\n")
file.write('        self.clf.fit(X,y)'+"\n")
file.write('        return self'+"\n")
file.write('    def predict(self, X):'+"\n")
file.write('        pr = [np.round(np.abs(item[0])) for item in self.clf.predict(X)]'+"\n")
file.write('        return pr'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        p_all.append([1-np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        p_all.append([np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()

from PLS import PLS

fasta = read_fasta('./input/seq.fasta')
pssm_composition = pd.read_csv('./input/pssm_composition.csv', header=None)
rpm_pssm = pd.read_csv('./input/rpm_pssm.csv', header=None)
s_fpssm = pd.read_csv('./input/s_fpssm.csv', header=None)

from joblib import load
raw_scl = load("./model/raw_scaler.sav")
feat_AAC = raw_scl[0].transform(AAC(fasta)[0])
feat_DPC = raw_scl[1].transform(DPC(fasta,0)[0])
feat_APAAC = raw_scl[2].transform(APAAC(fasta,1)[0])
feat_CTDC = raw_scl[3].transform(CTDC(fasta)[0])
feat_CTD = raw_scl[4].transform(np.hstack((CTDC(fasta)[0], CTDD(fasta)[0], CTDT(fasta)[0])))
feat_PSSM1 = raw_scl[5].transform(rearrange(fasta, pssm_composition))
feat_PSSM2 = raw_scl[6].transform(rearrange(fasta, rpm_pssm))
feat_PSSM3 = raw_scl[7].transform(rearrange(fasta, s_fpssm))

allclf = load("./model/allclfs.sav")
pr1 = allclf[0].predict_proba(feat_AAC)[:,0]
pr2 = allclf[1].predict_proba(feat_DPC)[:,0]
pr3 = allclf[2].predict_proba(feat_DPC)[:,0]
pr4 = allclf[3].predict_proba(feat_APAAC)[:,0]
pr5 = allclf[4].predict_proba(feat_APAAC)[:,0]
pr6 = allclf[5].predict_proba(feat_CTDC)[:,0]
pr7 = allclf[6].predict_proba(feat_CTDC)[:,0]
pr8 = allclf[7].predict_proba(feat_CTD)[:,0]
pr9 = allclf[8].predict_proba(feat_PSSM1)[:,0]
pr10 = allclf[9].predict_proba(feat_PSSM2)[:,0]
pr11 = allclf[10].predict_proba(feat_PSSM2)[:,0]
pr12 = allclf[11].predict_proba(feat_PSSM3)[:,0]

fname = ['AAC_LR', 'DPC_LN', 'DPC_PLS', 'APAAC_SVM', 'APAAC_RF', 'CTDC_LN',
         'CTDC_SVM', 'CTD_RF', 'pssm_composition_PLS', 'rpm_pssm_LR',
         'rpm_pssm_SVM', 's_fpssm_SVM']

allpr = np.hstack((pr1.reshape((len(pr1),1)),pr2.reshape((len(pr1),1)),pr3.reshape((len(pr1),1)),
                   pr4.reshape((len(pr1),1)),pr5.reshape((len(pr1),1)),pr6.reshape((len(pr1),1)),
                   pr7.reshape((len(pr1),1)),pr8.reshape((len(pr1),1)),pr9.reshape((len(pr1),1)),
                   pr10.reshape((len(pr1),1)),pr11.reshape((len(pr1),1)),pr12.reshape((len(pr1),1)),))

saphire_scl = load("./model/scalethemophilic.sav")
opt_f = np.array([ 1,  6,  8, 21, 22, 30, 33, 52, 56, 61, 63, 69])
tmp = np.zeros((len(allpr),72))
tmp[:,opt_f] = allpr
Xt = saphire_scl.transform(tmp)[:,opt_f]

saphire = load("./model/saphire.sav")
p_label = saphire.predict(Xt)
p_prob = saphire.predict_proba(Xt)[:,0]

label = ['Thermophilic', 'non-Thermophilic'] 
file = open("./output/predict_result.csv","w")
for i, (head, seq) in enumerate(fasta):
    file.write(head+","+label[int(p_label[i])]+","+str(p_prob[i])+"\n")
file.close()
