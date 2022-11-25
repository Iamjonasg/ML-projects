import whisper

model = whisper.load_model("large")
result = model.transcribe("dindon.mp3")
print(result["text"])

with open('readme.txt', 'w') as f:
    f.write(result["text"])