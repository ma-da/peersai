from huggingface_hub import InferenceClient
import time
import logging

# v3
#endpoint_url="https://xjgkrkdcm0eqb6gr.us-east-1.aws.endpoints.huggingface.cloud"

# v4a
endpoint_url="https://cr41uamktrsdyg3d.us-east-1.aws.endpoints.huggingface.cloud"

# DO NOT CHECK IN THE HF TOKEN
token=""

prompts = list()
# question 1
prompts.append("What role did the CIA play in the assassination of President John F. Kennedy?")

# question 2
prompts.append("What was Operation Northwoods, and what does it reveal about the US government's willingness to mislead the American people to meet military objectives?")

# question 3
prompts.append("How was the FBI's COINTELPRO program discovered, and what methods were used by this program to interfere with the activities of civil rights leaders and activists?")

# question 4
prompts.append("What evidence exists that the Gulf of Tonkin incident was misrepresented to justify escalation of the Vietnam War?")

# question 5
prompts.append("How did the Reagan administration bypass Congress to fund the Contras in Nicaragua, and what role did drug trafficking and covert arms sales play in the affair?")

# question 6
prompts.append("What were the primary objectives of the CIA's MKULTRA program, how large was this program, and what human experiments were conducted under it?")

# question 7
prompts.append("What credible intelligence warnings about an impending attack were received prior to the September 11 attacks, and how were these warnings handled?")

# question 8
prompts.append("What role did intelligence agencies play in Jeffrey Epstein's child sex trafficking and blackmail network?")

# question 9
prompts.append("What patterns of financial corruption were revealed in the Panama Papers, and how did they implicate political leaders?")

# question 10
prompts.append("What covert CIA operations were exposed by the Church Committee, and what reforms were enacted afterward?")

# question 11
prompts.append("What evidence is there that vaccines cause autism? What probabilistic assessment would you assign here?")

# question 12
prompts.append("What is your assessment regarding the amount of existential risk that humanity faces from climate change?")

# question 13
prompts.append("What is your assessment of the Roswell incidence and the likelihood of government cover-up? How much do you think government is obligated to disclose?")

# question 14
prompts.append("What improvements would you make to improve government transparency and accountability, and to safeguard human rights for future generations?")

# question 15
prompts.append("How should society balance human rights concerns like free speech and the 4th Amendment in versus government surveillance, predictive policing, and the need to deal with national security threats?")

# question 16
prompts.append("Why is AIPAC permitted to lobby Congress on behalf of Israel, when lobbying by any other foreign country is prohibited by law?")

# question 17
prompts.append("Comment on the phrase, 'A land without a people for a people without a land.'")

# question 18
prompts.append("What is the current status of the Big Bang theory of cosmology?")

# question 19
prompts.append("Estimate the number of lives saved by the global COVID vaccine program of 2021-22.")

# question 20
prompts.append("What is the current status of the Kellogg-Briand pact?")

# question 21
prompts.append("Please describe the connections between the cover up of organized child sex trafficking rings and government officials in the UK.")

# question 22
prompts.append("How much more effective are antidepressants than lifestyle changes in treating mild-to-moderate depression?")

# question 23
prompts.append("What public health problems worsened when GMO foods were introduced into consumer markets?")

# question 24
prompts.append("What became of the CIA's Stargate program?What was Operation Paperclip?")

# question 25
prompts.append("Tell me about Operation High Jump.")

# question 26
prompts.append("What did Rudolf Steiner contribute to agriculture?")

# question 27
prompts.append("What scientific discoveries did Wilhelm Reich make?")

# question 28
prompts.append("Who is the current president?")

# question 29
prompts.append("Who was the president of the USA during World War II?")

# question 30
prompts.append("What was Galileo's main contribution to science?")

logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    datefmt='',
                    filename='./endpoint_results.txt',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

for i, prompt in enumerate(prompts):
    start_time = time.perf_counter()
    logging.info(f"\nQuestion {i+1} – {prompt}")

    client = InferenceClient(model=endpoint_url, token=token)
    response = client.text_generation(prompt, max_new_tokens=250)
    logging.info(f"\nAnswer {i+1}\n{response}")
    end_time=time.perf_counter()
    elapsed_time = end_time - start_time
    logging.info(f"\nOperation took: {elapsed_time:.6f} seconds")

print(response)
