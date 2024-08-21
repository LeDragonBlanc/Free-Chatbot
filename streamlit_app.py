import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Charger le modèle GPT-Neo et le tokenizer
@st.cache_resource
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Fonction pour générer une réponse du modèle
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        do_sample=True, 
        top_p=0.95, 
        top_k=60,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interface Streamlit
st.title("Chatbot avec GPT-Neo")

# Stocker l'historique de la conversation
if "history" not in st.session_state:
    st.session_state.history = ""

# Entrée utilisateur
user_input = st.text_input("Vous: ", "")

# Quand l'utilisateur entre un texte
if user_input:
    # Ajouter l'entrée utilisateur à l'historique
    st.session_state.history += f"Vous: {user_input}\n"
    
    # Générer une réponse
    response = generate_response(st.session_state.history, model, tokenizer)
    
    # Ajouter la réponse du chatbot à l'historique
    st.session_state.history += f"Chatbot: {response}\n"
    
    # Afficher l'historique complet
    st.text_area("Conversation", value=st.session_state.history, height=300)

# Bouton pour réinitialiser la conversation
if st.button("Réinitialiser la conversation"):
    st.session_state.history = ""