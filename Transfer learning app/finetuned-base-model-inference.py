# 3. INFERENCE + PARAPHRASING + CHUNKING

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, textwrap

# Summarizer
sum_tokenizer = AutoTokenizer.from_pretrained("legal-summarizer-final", local_files_only=True)
sum_model     = AutoModelForSeq2SeqLM.from_pretrained("legal-summarizer-final",
                                                      local_files_only=True).to("cuda")

# Paraphraser (optional)
para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
para_model     = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to("cuda")

def generate_summary(text: str) -> str:
    inputs = sum_tokenizer(text, return_tensors="pt", max_length=1024,
                           truncation=True, padding="max_length").to(sum_model.device)
    ids = sum_model.generate(inputs["input_ids"], max_length=256,
                             num_beams=3, early_stopping=True)
    return sum_tokenizer.decode(ids[0], skip_special_tokens=True)

def paraphrase(text: str) -> str:
    prompt = f"paraphrase: {text} </s>"
    inputs = para_tokenizer(prompt, return_tensors="pt", max_length=512,
                            truncation=True, padding="max_length").to(para_model.device)
    ids = para_model.generate(inputs["input_ids"], max_length=256,
                              num_beams=3, early_stopping=True)
    return para_tokenizer.decode(ids[0], skip_special_tokens=True)

# ---------- Chunking helper (for >1024-token texts) ----------
def chunk_text(text: str, tok, max_tokens=900, overlap=100):
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks, n = [], len(ids)
    start = 0
    while start < n:
        end = min(start+max_tokens, n)
        chunk_ids = ids[start:end]
        chunks.append(tok.decode(chunk_ids, skip_special_tokens=True))
        start = end - overlap
    return chunks

def summarize_long_legal_text(legal_text: str) -> str:
    # Decide if we need chunking
    ids_len = len(sum_tokenizer(legal_text)["input_ids"])
    if ids_len <= 1024:
        raw = generate_summary(legal_text)
        return paraphrase(raw)

    # Otherwise chunk
    summaries = []
    for i, chunk in enumerate(chunk_text(legal_text, sum_tokenizer)):
        print(f"→ Chunk {i+1}")
        summaries.append(generate_summary(chunk))
    combined = " ".join(summaries)            # first merge…
    return paraphrase(combined)                # …then paraphrase

# ─────────── QUICK TEST ───────────
if __name__ == "__main__":
    sample = """
LESSEE shall be responsible for keeping the Vehicle and Equipment in good operating condition and working order, and making all necessary repairs, maintenance, and inspections to such Vehicle in accordance with the manufacturer’s suggested maintenance program.  LESSEE will comply with the manufacturer’s and LESSOR’s standards for vendors performing warranty and service work. Authorized direct dealers of the manufacturer shall be automatically deemed to comply.  LESSEE shall see that the Vehicle and Equipment are properly garaged, stored, and cleaned. 
LESSOR shall not be obligated to provide or pay for any maintenance or repair to the vehicle, or for any washing, parking, garaging, or other fees, tolls, fines, or liens of any nature that may be incurred in connection with the operation of the Vehicle and Equipment.  LESSEE shall indemnify and hold LESSOR harmless from any and all fines, penalties, and forfeitures imposed on account of the operation of the Vehicle in violation of any law or ordinance, together with expenses incurred by LESSOR in connection with the same. 
LESSOR shall have the right to inspect, adjust, or repair the Vehicle and Equipment at any reasonable time and place.  LESSOR agrees to cooperate fully to facilitate such inspection, adjustment, or repair.
LESSEE shall permit only licensed, qualified commercial motor vehicle drivers to operate the Vehicle and shall require them to operate the Vehicle with reasonable care and diligence and to comply with any standard written instructions issued by LESSOR covering the operation and maintenance of the Vehicle. All drivers used by LESSEE to operate the Vehicle under this Agreement must be duly qualified in accordance with the regulations of the FMCSA at 49 C.F.R. 391 et seq. Customer shall obtain and maintain in its business records the driver’s license number, state of issuance, and residence of each driver.
LESSEE shall not permit the Vehicle to be used in violation of any federal, state, or municipal statutes, laws, ordinances, or regulations, or contrary to the provisions of any applicable insurance policy.
LESSEE agrees not to overload the vehicle, operate it in violation of FMCSA’s safety regulations at 49 C.F.R. 392 et seq, or use the Vehicle outside the continental United States or Canada with LESSOR’s express written permission.
LESSOR shall have the right to substitute for any Vehicle or Equipment under this Agreement a Vehicle or Equipment of similar type and condition.  In the event of any substitution the term of this Agreement shall continue to be computed from the date of delivery of the original Vehicle. A substitution shall be at the sole discretion of LESSOR and LESSEE shall not have the right to a substitution.
In no event shall the Vehicle be operated by any person under the influence of alcohol, drugs, or any controlled substance. LESSOR may cancel this Agreement in the event that the Vehicle is so operated and may forbid any person found to be operating the vehicle in violation of this provision from driving the Vehicle again.
    """
    print("=========== EASY LEGAL SUMMARY ===========\n")
    print(textwrap.fill(summarize_long_legal_text(sample), width=90))
