import os
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import streamlit.components.v1 as components
from google import genai
from dotenv import load_dotenv

# --- 1. SECRETS & CONFIGURATION ---
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("API Key not found. Please set GEMINI_API_KEY in your .env file or Hugging Face secrets.")
        st.stop()

client = genai.Client(api_key=api_key)

# Register the custom MathLive component
mathlive_input = components.declare_component("mathlive_input", path="math_keyboard")

# --- 2. DATABASES ---
QUESTION_BANK = [
    {
        "id": "q_basic_1",
        "problem": "Differentiate $f(x) = 4x^3 - 2e^x + 9$",
        "function": "4*x**3 - 2*exp(x) + 9",
        "variable": "x",
        "skills": ["Power Rule", "Exponential Rule", "Constant Rule", "Sum/Difference Rule"]
    },
    {
        "id": "q_basic_2",
        "problem": "Differentiate $g(x) = \\frac{6}{x} + 5x$",
        "function": "6/x + 5*x",
        "variable": "x",
        "skills": ["Power Rule with Negative Exponents", "Power Rule"]
    },
    {
        "id": "q_basic_3",
        "problem": "Differentiate $f(x) = 4\\sqrt{x} + 3x^2 - e^x$",
        "function": "4*sqrt(x) + 3*x**2 - exp(x)",
        "variable": "x",
        "skills": ["Power Rule with Fractional Exponents", "Power Rule", "Exponential Rule"]
    }
]

STUDENT_PROFILES = {
    "STU-101 (Beginner)": {
        "student_id": "STU-101",
        "proficiency": "Beginner",
        "recurring_errors": ["Leibniz false start (u'v' instead of u'v + uv')", "Sign errors with negative exponents"]
    },
    "STU-202 (Careless Intermediate)": {
        "student_id": "STU-202",
        "proficiency": "Intermediate",
        "recurring_errors": ["Rushes through algebra, often dropping constants or coefficients", "Forgets to distribute negative signs"]
    },
    "STU-303 (Conceptual Learner)": {
        "student_id": "STU-303",
        "proficiency": "Beginner",
        "recurring_errors": ["Struggles with abstract notation", "Needs real-world analogies to understand the rules"]
    }
}

# --- 3. SYMPY VERIFICATION TOOL ---
def verify_robust_derivative(function_str, student_math, variable='x'):
    """Parses standard math text with automatic implicit multiplication correction."""
    try:
        # Enable rules to auto-fix "2x" to "2*x" and "x^2" to "x**2"
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        
        var = sp.Symbol(variable)
        original_function = sp.sympify(function_str)
        
        # FIX: Add local_dict={"e": sp.E} to translate the letter 'e' into the mathematical constant
        student_derivative = parse_expr(
            student_math, 
            transformations=transformations,
            local_dict={"e": sp.E}
        )
        true_derivative = sp.diff(original_function, var)
        
        is_correct = true_derivative.equals(student_derivative)
        return is_correct, None
    except Exception as e:
        return False, "Parse Error"

# --- 4. GEMINI PEDAGOGICAL ENGINE ---
def generate_hint(question, profile, student_math, student_text):
    """Prompts Gemini to diagnose the error and scaffold learning."""
    prompt = f"""
    You are an expert AI Calculus Tutor interacting with a student in a chat. 
    The student has submitted an incorrect answer. Diagnose the error and provide a scaffolding hint. 
    
    CRITICAL RULE: DO NOT GIVE THE FINAL ANSWER OR THE FULL CORRECT EQUATION. 
    
    === SYSTEM CONTEXT ===
    Problem: {question['problem']}
    Correct Mathematical Function: {question['function']}
    
    === LEARNER PROFILE ===
    Proficiency: {profile['proficiency']}
    Known Recurring Errors: {', '.join(profile['recurring_errors'])}
    
    === STUDENT INPUT ===
    Student's Incorrect Math (ASCII format): {student_math}
    Student's Explanation: {student_text}
    
    === INSTRUCTIONS ===
    1. Speak directly to the student in a warm, encouraging, and conversational tone.
    2. Read their explanation to deduce their misconception.
    3. Generate a concise hint (1-3 sentences) that guides them to the next logical step.
    4. Keep it brief and conversational, suitable for a chat interface.
    """
    
    try:
        # The new generate_content syntax requires passing the model name here
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"GEMINI API ERROR: {e}") 
        return f"System error generating hint."

# --- 5. SESSION STATE MANAGEMENT ---
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "solved" not in st.session_state:
    st.session_state.solved = False
if "messages" not in st.session_state:
    first_q = QUESTION_BANK[0]["problem"]
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! Let's get started on your calculus practice. \n\n**{first_q}**\n\nWhat is your first step?"}
    ]

current_q = QUESTION_BANK[st.session_state.q_index]

# --- 6. USER INTERFACE & WORKFLOW ---
st.set_page_config(page_title="AI Math Tutor", layout="centered")

# Interactive Sidebar Profile Selector
with st.sidebar:
    st.header("Control Panel")
    st.markdown("Use this to test how the AI adapts its hints to different student types.")
    
    selected_profile_key = st.selectbox("Select Active Learner Profile:", list(STUDENT_PROFILES.keys()))
    active_profile = STUDENT_PROFILES[selected_profile_key]
    
    st.subheader("Profile Data Sent to AI:")
    st.json(active_profile)

st.title("💬 Calculus ITS Chat")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- WORKFLOW STATE A: Not Solved ---
if not st.session_state.solved:
    
    st.markdown("---")
    
    with st.form("chat_composer"):
        st.markdown("**1. Enter your mathematical answer:**")
        
        # Bi-directional MathLive keyboard input (returns ASCII math)
        student_math = mathlive_input(key="math_input")
        
        st.markdown("**2. Explain your reasoning:**")
        student_text = st.text_area("Why did you take this step?", label_visibility="collapsed", placeholder="I used the product rule...")
        
        submit_btn = st.form_submit_button("Send to Tutor")
        
        if submit_btn:
            if not student_math or not student_text:
                st.warning("Please provide both the math and your explanation.")
            else:
                # Add user message to chat
                user_msg = f"**My Answer:** `{student_math}`\n\n**My Reasoning:** {student_text}"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                
                # Verify with SymPy robust parser
                is_correct, error = verify_robust_derivative(current_q["function"], student_math, current_q["variable"])
                
                # Generate Tutor Response
                if is_correct:
                    st.session_state.solved = True
                    tutor_msg = "Excellent! Your mathematical reasoning and the final derivative are perfectly correct. 🎉"
                    st.session_state.messages.append({"role": "assistant", "content": tutor_msg})
                else:
                    st.session_state.solved = False
                    if error == "Parse Error":
                        tutor_msg = "I couldn't parse that mathematical format. Please ensure your expression is complete before submitting."
                    else:
                        with st.spinner("Tutor is typing..."):
                            tutor_msg = generate_hint(current_q, active_profile, student_math, student_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": tutor_msg})
                
                st.rerun()

# --- WORKFLOW STATE B: Solved ---
else:
    st.markdown("---")
    if st.session_state.q_index < len(QUESTION_BANK) - 1:
        if st.button("Move to Next Problem ➡️"):
            st.session_state.q_index += 1
            st.session_state.solved = False
            next_q = QUESTION_BANK[st.session_state.q_index]["problem"]
            
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Great job on that one. Let's move to the next challenge:\n\n**{next_q}**\n\nWhat's your approach?"}
            )
            st.rerun()
    else:
        st.success("🎉 You have completed all the practice problems for this session!")
        if st.button("Restart Session"):
            st.session_state.clear()
            st.rerun()