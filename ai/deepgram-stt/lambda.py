from deepgram import Deepgram

import json
import io
import cgi
import os
import uuid
import base64

DEEPGRAM_API_KEY = os.environ['DEEPGRAM_API_KEY']

def lambda_handler(event, context):
    
    print(event)
    
    # change multipart/form-data to file
    fp = io.BytesIO(base64.b64decode(event['body']+ "========"))
    _, pdict = cgi.parse_header(event['headers']['content-type'])
    pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')

    form_data = cgi.parse_multipart(fp, pdict)
    
    print(form_data)
    
    audio_data = form_data['data'][0]
    
    print(f"Retrieved audio data with boundary: {pdict['boundary']}, audio length: {len(audio_data)}")

    temp_file = f'/tmp/{str(uuid.uuid4())}'
    with open(temp_file, 'wb') as f:
        f.write(audio_data)
    print(f"Audio data saved at: {temp_file}")
    
    # Initializes the Deepgram SDK
    deepgram = Deepgram(DEEPGRAM_API_KEY)

    response = ''
    
    # Open the audio file
    with open(temp_file, 'rb') as audio:
        # ...or replace mimetype as appropriate
        source = {'buffer': audio, 'mimetype': 'audio/mp3'}
        response = deepgram.transcription.sync_prerecorded(source, {'punctuate': True})
    
    os.remove(temp_file)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'transcription': response
        })
    }
