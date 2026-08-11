import json
import os

os.makedirs('training/data', exist_ok=True)

samples = [
    {
        "query": "Which magazine was started first Arthur's Magazine or First for Women?",
        "evidence": "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        "synthesized": "Arthur's Magazine was started first."
    },
    {
        "query": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "evidence": "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.The Oberoi Group is a hotel company with its head office in Delhi.",
        "synthesized": "The Oberoi Group has its head office in Delhi."
    },
    {
        "query": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
        "evidence": "Allison Beth \"Allie\" Goertz (born March 2, 1991) is an American musician. Goertz is known for her satirical songs based on various pop culture topics. Her videos are posted on YouTube under the name of Cossbysweater.Milhouse Mussolini van Houten is a fictional character featured in the animated television series \"The Simpsons\", voiced by Pamela Hayden, and created by Matt Groening who named the character after President Richard Nixon's middle name.",
        "synthesized": "Matt Groening named Milhouse after President Richard Nixon's middle name."
    },
    {
        "query": "What nationality was James Henry Miller's wife?",
        "evidence": "Margaret \"Peggy\" Seeger (born June 17, 1935) is an American folksinger. She is also well known in Britain, where she has lived for more than 30 years, and was married to the singer and songwriter Ewan MacColl until his death in 1989.James Henry Miller (25 January 1915 - 22 October 1989), better known by his stage name Ewan MacColl, was an English folk singer, songwriter, communist, labour activist, actor, poet, playwright and record producer.",
        "synthesized": "James Henry Miller's wife was American."
    },
    {
        "query": "Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?",
        "evidence": "Jonathan Stark (born April 3, 1971) is a former professional tennis player from the United States. During his career he won two Grand Slam doubles titles (the 1994 French Open Men's Doubles and the 1995 Wimbledon Championships Mixed Doubles). He reached the men's singles final at the French Open in 1988, won the French Open men's doubles title in 1984, and helped France win the Davis Cup in 1991.",
        "synthesized": "The evidence states that Jonathan Stark won two Grand Slam titles, but does not provide information on Henri Leconte."
    },
    {
        "query": "Which genus of moth in the world's seventh-largest country contains only one species?",
        "evidence": "Indogrammodes is a genus of moths of the family Crambidae. It contains only one species, Indogrammodes pectinicornalis, which is found in India.India, officially the Republic of India (Bharat Ganarajya), is a country in South Asia. It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world.",
        "synthesized": "The Indogrammodes genus of moth contains only one species."
    },
    {
        "query": "Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring.",
        "evidence": "Fighters from around world on the roster include Badr Hari, Peter Aerts, Peter Graham, Dewey Cooper, Zabit Samedov. It was considered as one of the biggest kickboxing and MMA promotion in Middle East.Badr Hari (Arabic: بدر هاري ; born 8 December 1984) is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike's Gym in Oostzaan. Hari has been a prominent figure in the world of kickboxing and was once considered the best kickboxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring.",
        "synthesized": "Badr Hari was once considered the best kickboxer in the world but has been involved in controversies regarding unsportsmanlike conduct and crimes of violence."
    },
    {
        "query": "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?",
        "evidence": "The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour.Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.",
        "synthesized": "The track is 6.213 km long."
    },
    {
        "query": "Fast Cars, Danger, Fire and Knives includes guest appearances from which hip hop record executive?",
        "evidence": "Vocals are handled by Aesop Rock, with guest appearances from Camu Tao and Metro of S.A. Smash and Definitive Jux label head El-P.Jaime Meline (born March 2, 1975), better known by his stage name El-P (shortened from El Producto), is an American hip hop recording artist, record producer, and record executive.",
        "synthesized": "The guest appearance is from El-P."
    },
    {
        "query": "Gunmen from Laredo starred which narrator of \"Frontier\"?",
        "evidence": "Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy.Walter Darwin Coy (January 31, 1909 - December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana. He was best known for narrating the NBC western anthology series, \"Frontier\", which aired early Sunday evenings in the 1955-1956 season.",
        "synthesized": "It starred Walter Coy."
    }
]

with open('training/data/shape_b_llm_synthesized_SAMPLE.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
print("Wrote sample to training/data/shape_b_llm_synthesized_SAMPLE.jsonl")
