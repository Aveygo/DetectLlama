#from detect_fast import DetectLlama
from detect_slow import DetectLlama

detect = DetectLlama()

text = """
As the sun began to set over the small town of Millfield, Sarah walked along the quiet streets with a heavy heart. 
She had just received the news that her grandmother had passed away and she was struggling to come to terms with it. 
Her grandmother had been her rock throughout her life, always there to offer words of wisdom and
a comforting hug when she needed it the most. 
As Sarah passed by the old bookstore on Main Street, she paused for a moment and peered through the window. 
The store had been closed for as long as she could remember, but she had always been fascinated
by the dusty old books that lined the shelves. As she gazed inside, she noticed a flicker of movement in the shadows.
Curiosity getting the best of her, Sarah pushed open the creaky door and stepped inside. 
The musty smell of old books filled her nostrils as she made her way down the narrow aisle,
her eyes scanning the shelves for any sign of movement. Suddenly, she heard a soft rustling sound coming from the 
back of the store. Her heart racing with excitement, Sarah tiptoed towards the sound, eager to discover 
what secrets lay hidden within the dusty old bookstore."""

results = []

for i, ppl in enumerate(detect.calculate(text)):
    text = ppl["text"]
    score = ppl["score"]
    results.append(score)
    print(ppl)

print(f"Final score: {sum(results)/len(results)}") 