import numpy as np
import pandas as pd
import json
import os
from scipy.signal import resample
import csv
import scipy
import librosa
from datetime import datetime
# import soundfile as sf

annotation_filepath = "/work/hdd/bebr/Data/EMA_Annotations/BRP1/"
annotaion_files = []
for dirpath, dirnames, filenames in os.walk(annotation_filepath):
    for fname in filenames:
        if fname.endswith(".txt"):
            full_path = os.path.join(dirpath, fname)
            annotaion_files.append(full_path)

import numpy as np 
import pandas as pd 
import os 
from datetime import datetime, timedelta


def convert_edited_elan(filepath):
    current_time = datetime.now()

    # filepath = "/Users/mkhan/Littlebeats/FOR NUR_EMA Conversion For ELAN-selected/EDITED_EMA TXT files (Please convert to 1s CSVs for reliability comparison)/BRP1_ID25003_Obs5A_02-03-2025_HF_EDITED.txt"

    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['Tier', 'nan', 'start', 'r1', 'end', 'Duration', 'r2', 'r3', 'label']
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df.drop(['nan', 'r1', 'r2', 'r3'], axis=1, inplace=True)
    # print(df.head())
    # save_name = 'ELAN_' + filepath.split('/')[-1][:-4]+'.csv'
    # df.to_csv(save_name)

    return df

        

tone_detection_file = "/work/hdd/bebr/Data/Timeinfo/BRP1_ToneDetection/BRP1_Tone_Alignment_Nur_3-10-2026.xlsx"# vv_tone_detection_file = "/work/hdd/bebr/Data/LB/timeinfo/2_BCP_VirtualVisit_tone-detection/ToneTime_Baseline-VirtualVisit_Cleaned-2025-09-30.xlsx"
df_tone = pd.read_excel(tone_detection_file)

print(len(df_tone))
# def get_tone_value(sitename, id, site_type):

#     tone_value = df_tone.loc[(df_tone["subject_id"] == int(id)) & (df_tone["sitename"] == sitename) & (df_tone["timepoint"]==site_type), "start_time_s"]
#     try:
#         tone_value = float(tone_value)
#     except:
#         return None
#     return tone_value

def read_directories(root_dir):
    ecg_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # print("Current directory:", dirpath)

        # for dirname in dirnames:
        #     print("Subdirectory:", os.path.join(dirpath, dirname))

        for filename in filenames:
            
            if filename.endswith('ecg.wav'):
                ecg_files.append(os.path.join(dirpath, filename))
            
    return ecg_files
# label_root_dir = "/work/hdd/bebr/Data/Behavioral_Coding/BCP_Baseline_InfantStates_LabVisit"
# all_label_file = os.listdir(label_root_dir)
# print(len(all_label_file))
# def get_label_file(sitename, id, site_type):
#     for file in all_label_file:
#         l_sitename = file.split('_')[0].split('-')[-1]
#         l_id = int(file.split('_')[1][2:])
#         l_sitetype = file.split('_')[2]
#         if l_sitename==sitename and l_id==int(id) and l_sitetype==site_type:
#             return file
#     return None


file_root_dir = "/work/hdd/bebr/Data/LB/ecg_v2/BRP1" # Replace with the actual path
ecg_files = read_directories(file_root_dir)
print(len(ecg_files))

print(len(annotaion_files))
tone_filenames = df_tone['file_name'].tolist()
observations = df_tone['obs_number'].tolist()
tone_sec = df_tone['tone_time_seconds'].tolist()
final_annotaion_files = []
final_ecg_files = []
tone_sec_final = []
for tone_file, obs, tone in zip(tone_filenames, observations, tone_sec):
    tone_split = tone_file.split('/')[-1].split('_')

    
    for label_file in annotaion_files:
        label_split = label_file.split('/')[-1].split('_')
        label_id = label_split[1]
        label_obs = label_split[2][3:]
        # print(label_id, label_obs, obs, tone_split[1])
        if label_id==tone_split[1] and label_obs == obs:
            final_annotaion_files.append(label_file)
            # print("label: ", label_split)
            tone_sec_final.append(tone)
            for idx, ecg_file in enumerate(ecg_files):
                ecg_split = ecg_file.split('/')[-1].split("_")
                if tone_split[1]==ecg_split[1] and tone_split[3]==ecg_split[2]:
                    final_ecg_files.append(ecg_file)
                    break
            break



print(len(final_ecg_files),len(final_annotaion_files), len(tone_sec_final) )
df = pd.concat([pd.DataFrame({"ECG_files": final_ecg_files}), pd.DataFrame({"label_file": final_annotaion_files}), pd.DataFrame({"tone_sec": tone_sec_final})], axis = 1)
df.to_csv("BRP_annotated_files_with_tones.csv")
exit()




# sr=70
# count=0

        
def parse_imu_txt(filename):
    all_numbers=[]
    with open(filename, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            # Convert each line to a NumPy array of floats
            nums = np.fromstring(line, sep=' ')
            all_numbers.append(nums)
    # print(len(all_numbers))
    if len(all_numbers)>0:
        all_numbers = np.vstack(all_numbers)
    
    return all_numbers

test_imu_chunks = []
test_imu_labels = []
train_imu_chunks = []
train_imu_labels = []
test_ids = [25008, 25004]
def reshape_data(chunk, target_len=240):
    chunk = resample(chunk, target_len, axis=1)
    return chunk
def make_chunks(start, end, label, offset, imu_file, id_, save_dir, sr=70, duration=3):
    global count_train
    global count_test
    start = time_to_seconds(start)+offset
    end = time_to_seconds(end)+offset
    full_chunk = imu_file[:, int(start*sr):int(end*sr)]
    for i in range(0, int(end-start), duration):
        if (i+duration)*sr>full_chunk.shape[1]:
            break
        chunk = full_chunk[:, i*sr:int((i+duration)*sr)]
        chunk = reshape_data(chunk)
        if int(id_) in test_ids:
            save_name = save_dir+'test/'+str(count_test)+"_"+str(label)+".npy"
            np.save(save_name, chunk)
            count_test+=1
            # test_imu_chunks.append(chunk)
            # test_imu_labels.append(label)
        else:
            save_name = save_dir+'train/'+str(count_train)+"_"+str(label)+".npy"
            np.save(save_name, chunk)
            count_train+=1
            # train_imu_chunks.append(chunk)
            # train_imu_labels.append(label)

save_dir = "/work/hdd/bebr/jiaxuan_abstract/BRP_audio_chunk/"
def make_chunks_audio(start, end, label, offset, audio_file, id, sr=16000, duration=3):
    global count_train
    global count_test
    global data_dict_tr
    global data_dict_te
    start = time_to_seconds(start)+offset
    end = time_to_seconds(end)+offset
    full_chunk = audio_file[int(start*sr):int(end*sr)]
    for i in range(0, int(end-start), duration):
        if (i+duration)*sr>full_chunk.shape[0]:
            break
        chunk = full_chunk[i*sr:int((i+duration)*sr)]
        # save_name = save_dir+str(count_train)+"_"+str(label)+".wav"
        # scipy.io.wavfile.write(save_name, sr, chunk)
        # count_train+=1

        
        if int(id) in test_ids:
            save_name = save_dir+'test/'+str(count_test)+"_"+str(label)+".wav"
            scipy.io.wavfile.write(save_name, sr, chunk)
            count_test+=1
            data_dict = {}
            data_dict["labels"] = ''
            data_dict["wav"] = save_name
            data_dict['labels'] = data_dict["labels"]+'/m/'+str(label)
            data_dict_te.append(data_dict)
        else:
            save_name = save_dir+'train/'+str(count_train)+"_"+str(label)+".wav"
            scipy.io.wavfile.write(save_name, sr, chunk)
            count_train+=1
            data_dict = {}
            data_dict["labels"] = ''
            data_dict["wav"] = save_name
            data_dict['labels'] = data_dict["labels"]+'/m/'+str(label)
            data_dict_tr.append(data_dict)

def time_to_seconds(t):
    dt = datetime.strptime(t, "%H:%M:%S.%f")
    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    return seconds

def make_chunks_lb_llm(start, end, label, offset, audio_file, id_, sr=16000, duration=10):
    global count_train
    global count_test
    global data_dict_tr
    global data_dict_te
    skip=0.1
    start = time_to_seconds(start)+offset
    end = time_to_seconds(end)+offset
    full_chunk = audio_file[int(start*sr):int(end*sr)]
    start_chunk = 0
    while (start_chunk+duration)*sr<full_chunk.shape[0]:
        chunk = full_chunk[int(start_chunk*sr):int((start_chunk+duration)*sr)]
        start_chunk+=skip
        # save_name = save_dir+str(count_train)+"_"+str(label)+".wav"
        # scipy.io.wavfile.write(save_name, sr, chunk)
        # count_train+=1

        
        if int(id_) in test_ids:
            save_name = save_dir+'chunk_for_lb_llm/'+str(count_test)+"_"+str(label)+".wav"
            scipy.io.wavfile.write(save_name, sr, chunk)
            data_dict_te[f"id{count_test+1}"] = {
                "wav": {"file": save_name},
                "label": label
            }
            count_test+=1

        else:
            continue
quiet_list = ['quiet alert', 'light sleep', 'deep sleep', 'drowsy', 'drowsy unsure']
tone = df_tone['tone_time_seconds'].tolist()
count_test=0
count_train=0
data_dict_tr = []
data_dict_te = {}
for idx, label_file in enumerate(final_annotaion_files):
    df = convert_edited_elan(label_file)
    df = df[df["Tier"]=='state']
    imu_file = final_imu_files[idx]
    id_ = imu_file.split('/')[-1].split('_')[1]
    offset = tone[idx]
    imu_file = parse_imu_txt(imu_file)
    audio_file, sr = librosa.load(final_audio_files[idx], sr=None)
    for idx, row in df.iterrows():
        start = row['start']
        end = row['end']
        label = row['label']
        if label in quiet_list:
            label = 0
        else:
            label = 1
        # make_chunks(start=start, end=end, label=label, offset=offset,save_dir="/work/hdd/bebr/jiaxuan_abstract/BRP_imu_chunk/", imu_file=imu_file, id_=id_)
        # make_chunks_audio(start=start, end=end, label=label, offset=offset, audio_file=audio_file, id=id_)
        make_chunks_lb_llm(start=start, end=end, label=label, offset=offset, audio_file=audio_file, id_=id_)
        # print(count_train)

with open("BRP_classroom_data_for_lb_testing.json", "w") as f:
    json.dump(data_dict_te, f, indent=4)
# final_dict_tr = {}
# final_dict_tr["data"] = data_dict_tr
# json_object_tr = json.dumps(final_dict_tr, indent=1)
# with open("/work/hdd/bebr/jiaxuan_abstract/ast/egs/BRP/data/datafiles/brp_train_data.json", "w") as outfile:
#     outfile.write(json_object_tr)

# final_dict_te = {}
# final_dict_te["data"] = data_dict_te 
# json_object_te = json.dumps(final_dict_te, indent=1)
# with open("/work/hdd/bebr/jiaxuan_abstract/ast/egs/BRP/data/datafiles/brp_eval_data.json", "w") as outfile:
#     outfile.write(json_object_te)

# test_imu_chunks = np.array(test_imu_chunks)
# test_imu_labels = np.array(test_imu_labels)
# train_imu_chunks = np.array(train_imu_chunks)
# train_imu_labels = np.array(train_imu_labels)
# np.save("/work/hdd/bebr/jiaxuan_abstract/BRP_imu_chunk/test_imu_chunk.npy", test_imu_chunks)
# np.save("/work/hdd/bebr/jiaxuan_abstract/BRP_imu_chunk/test_imu_label.npy", test_imu_labels)
# np.save("/work/hdd/bebr/jiaxuan_abstract/BRP_imu_chunk/train_imu_chunk.npy", train_imu_chunks)
# np.save("/work/hdd/bebr/jiaxuan_abstract/BRP_imu_chunk/train_imu_label.npy", train_imu_labels)
# print(test_imu_chunks.shape)
# print(test_imu_labels.shape)
# print(train_imu_chunks.shape)
# print(train_imu_labels.shape)
print(count_test, count_train)


