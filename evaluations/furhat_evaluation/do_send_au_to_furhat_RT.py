#! /opt/python-env/bin/python3.12

from furhat_realtime_api import FurhatClient
import asyncio

import numpy as np
from os import path, system
from sys import argv
from math import degrees

import logging
import random
import time
import colorsys

#~ import argparse

web_prefix= "http://192.168.70.52:8000/tts/"
file_prefix= "/localdata/httpd/html/tts/"

furhat= None
#furhat_async= None

#output, 2 + 49 +3 parameters
# AU_names="BROW_DOWN_LEFT BROW_DOWN_RIGHT BROW_INNER_UP BROW_OUTER_UP_LEFT BROW_OUTER_UP_RIGHT CHEEK_PUFF CHEEK_SQUINT_LEFT CHEEK_SQUINT_RIGHT \
AU_names="BROW_INNER_UP BROW_OUTER_UP_LEFT BROW_OUTER_UP_RIGHT CHEEK_PUFF CHEEK_SQUINT_LEFT CHEEK_SQUINT_RIGHT \
EYE_BLINK_LEFT EYE_BLINK_RIGHT EYE_LOOK_DOWN_LEFT EYE_LOOK_DOWN_RIGHT EYE_LOOK_IN_LEFT EYE_LOOK_IN_RIGHT EYE_LOOK_OUT_LEFT \
EYE_LOOK_OUT_RIGHT EYE_LOOK_UP_LEFT EYE_LOOK_UP_RIGHT EYE_SQUINT_LEFT EYE_SQUINT_RIGHT \
EYE_WIDE_LEFT EYE_WIDE_RIGHT JAW_FORWARD JAW_LEFT JAW_OPEN JAW_RIGHT MOUTH_CLOSE MOUTH_DIMPLE_LEFT MOUTH_DIMPLE_RIGHT \
MOUTH_FROWN_LEFT MOUTH_FROWN_RIGHT MOUTH_FUNNEL MOUTH_LEFT MOUTH_LOWER_DOWN_LEFT MOUTH_LOWER_DOWN_RIGHT \
MOUTH_PRESS_LEFT MOUTH_PRESS_RIGHT MOUTH_PUCKER MOUTH_RIGHT MOUTH_ROLL_LOWER MOUTH_ROLL_UPPER \
MOUTH_SHRUG_LOWER MOUTH_SHRUG_UPPER MOUTH_SMILE_LEFT MOUTH_SMILE_RIGHT MOUTH_STRETCH_LEFT MOUTH_STRETCH_RIGHT \
MOUTH_UPPER_UP_LEFT MOUTH_UPPER_UP_RIGHT NOSE_SNEER_LEFT NOSE_SNEER_RIGHT NECK_PAN NECK_TILT NECK_ROLL"

# Furhat needs 16kHz mono wav files
def mk_16k_wav(wav_src, wav_file):
    cmd=f"sox {wav_src} -r 16000 -c 1 -b 16 {wav_file}"
    print("CREATING "+wav_file)
    system(cmd)

# use the "articulate" format from the gipsa_dev/_furhat_API
def mk_txt_from_au(au_src, txt_name, opt_drop_last=None):
    print("CREATING "+txt_name)
    sel=None
    with open(au_src, "rb") as fid:
        header= np.fromfile(fid, dtype=np.int32, count=4)
        nb_samples, nb_au, freq_num, freq_div =header
        AU = np.fromfile(fid, dtype=(np.float32, nb_au))

    with open(txt_name,"wt") as fid:
      my_AU=AU_names.split(" ")[:opt_drop_last]
      print(" ".join(my_AU), file=fid)
      ts_div=0
      for i in range(AU.shape[0]):
          ts_div+=freq_div # inform the reader of the frame rate
          #sel=AU[i,3:opt_drop_last]
          sel=AU[i,5:opt_drop_last]
          print(f"{ts_div / freq_num:.3f}", *(f"{x:.3f}" for x in sel), file=fid)

      if len(sel)!=len(my_AU):
          print("Sizes don't match !!!")
          print(len(sel))
          print(len(my_AU))
          exit(-1)
      else:
        print(freq_num/freq_div, header, AU.shape, " -->",len(sel))

def disconnect_RT(server= None):
    global furhat

    if server and server!=furhat:
        print("Not the server that was registred!")

    if furhat:
        furhat.disconnect()
        print("Releasing Furhat RT API")
        furhat=None

def connect_RT(host="furhat"):
    global furhat
    global furhat_async

    furhat = FurhatClient(host, auth_key=None)
    furhat.set_logging_level(logging.INFO) # Change to logging.DEBUG to see all events

    try:
        furhat_async = furhat.async_client
        furhat.connect()
    except Exception as e:
        print(f"Failed to connect to Furhat on {args.host}.")
        exit(0)

    furhat.request_attend_location(0,0,1)

    return furhat

def do_send(index4, txt_file=None, wav_url=None):

    if txt_file is None:
        if index4 is not None:
            basename= f"INPUT_{int(index4):04}_syn"
        else:
            basename=txt_file
        wav_url= web_prefix+basename+".wav"
        txt_file= file_prefix+basename+".txt"
        wav_file= file_prefix+basename+".wav"
        if not path.isfile(wav_file) or not path.isfile(txt_file):
            print("At least one missing file, either",txt_file,wav_file,sep="\n    ")
            return []

    with open(txt_file) as fid:
      names=next(fid).strip().split()
      #~ print(names)
      uncached_wav_url = f"{wav_url}?t={int(time.time())}"
      furhat.request_speak_audio(url=uncached_wav_url, wait= False, abort= True, text= "TTS_text", lipsync= False)
      #~ system(f"aplay {src}.wav &")
      start=time.time()
      for i,line in enumerate(fid):
        fields=line.split()
        bs={ n: float(e) for n,e in zip(names[:-3],fields[1:-3]) }
        furhat.request_face_params(bs)
        bs={ n: (float(e)) for n,e in zip(names[-3:],fields[-3:]) }
        furhat.request_face_headpose(-bs["NECK_PAN"],-bs["NECK_TILT"], bs["NECK_ROLL"], False)
        #~ delay=start+int(fields[0])/1000.0 - time.time()
        delay=start+float(fields[0]) - time.time()
        #~ print(i, delay)
        if delay>0:
            time.sleep(delay)

      # asyncio.run( furhat_async.send_event({"type": "response.speak.end", "text": "hello"}))

def conv_and_send_to_furhat(index4, opt_drop_last=3):
    basename= f"INPUT_{int(index4):04}_syn"
    wav_url= web_prefix+basename+".wav"
    wav_file= file_prefix+basename+".wav"
    txt_file= file_prefix+basename+".txt"

    au_src=f"_syn_FURHAT_BS_86.1328125/"+basename+".au"
    wav_src=f"_syn_WAVEGLOW/"+basename+".wav"

    mk_16k_wav(wav_src, wav_file)
    mk_txt_from_au(au_src, txt_file, opt_drop_last)

    do_send(index4, txt_file, wav_url)

    return [wav_file, txt_file] # tmp files

#~ parser = argparse.ArgumentParser()
#~ parser.add_argument("--host", type=str, default="furhat", help="Furhat robot IP address")
#~ parser.add_argument("--auth_key", type=str, default=None, help="Authentication key for Realtime API")
#~ args = parser.parse_args()

# -- main --
if __name__=="__main__":

    furhat=connect_RT()

    exe=argv.pop(0)
    if len(argv)==1:
      src=argv.pop(0)
      do_send(src, None)
    elif len(argv)==2:
      where=argv.pop(0)
      basename=argv.pop(0)
      if basename.endswith(('.wav','.txt')):
        basename=basename[:-4]
      wav_ref= f"http://192.168.70.52:8000/{where}/{basename}.wav"
      file_ref= f"/localdata/httpd/html/{where}/{basename}.txt"
      do_send(None, file_ref, wav_ref    )

    print("Done")
    disconnect_RT(furhat)
