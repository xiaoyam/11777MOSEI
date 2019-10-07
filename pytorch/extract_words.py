import os

def main():
    id_f = open("all_ids.txt", 'w')
    words_f = open("all_words.txt", 'w')
    for text in os.listdir("../Transcript/Combined"):
        trans_doc = open("../Transcript/Combined/"+text, 'r')
        sentences = trans_doc.readlines()
        for sent in range(len(sentences)):
            seg = sentences[sent]
            lastThree = seg.rfind('___')
            transcript = seg[lastThree+3:]
            words_f.write(transcript)
            id_f.write(text[:-4] + "_" + str(sent) + '\n')


if __name__ == '__main__':
    main()
