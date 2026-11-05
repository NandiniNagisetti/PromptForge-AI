
import os
import re
import uuid
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Prompt Sensitivity Lab",
    page_icon="🧪",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "experiment_history.csv")


# =========================================================
# 2. BENCHMARK DATASET
# =========================================================


BENCHMARK_QUESTIONS = [
    # Geography
    {"question": "What is the capital of France?", "expected": "Paris", "category": "Geography"},
    {"question": "What is the capital of Japan?", "expected": "Tokyo", "category": "Geography"},
    {"question": "What is the capital of Australia?", "expected": "Canberra", "category": "Geography"},
    {"question": "What is the capital of Italy?", "expected": "Rome", "category": "Geography"},
    {"question": "What is the capital of Canada?", "expected": "Ottawa", "category": "Geography"},
    {"question": "What is the largest ocean on Earth?", "expected": "Pacific Ocean", "category": "Geography"},
    {"question": "Which is the largest continent by area?", "expected": "Asia", "category": "Geography"},
    {"question": "Which country is home to the Great Pyramid of Giza?", "expected": "Egypt", "category": "Geography"},
    {"question": "What is the capital of India?", "expected": "New Delhi", "category": "Geography"},
    {"question": "Which river flows through London?", "expected": "Thames", "category": "Geography"},

    # Mathematics
    {"question": "What is 12 multiplied by 8?", "expected": "96", "category": "Mathematics"},
    {"question": "What is the square of 15?", "expected": "225", "category": "Mathematics"},
    {"question": "What is the square root of 144?", "expected": "12", "category": "Mathematics"},
    {"question": "What is 125 plus 275?", "expected": "400", "category": "Mathematics"},
    {"question": "What is 1000 minus 365?", "expected": "635", "category": "Mathematics"},
    {"question": "What is 20 percent of 250?", "expected": "50", "category": "Mathematics"},
    {"question": "How many degrees are in a right angle?", "expected": "90", "category": "Mathematics"},
    {"question": "What is 7 factorial?", "expected": "5040", "category": "Mathematics"},
    {"question": "What is the perimeter of a square with side length 5?", "expected": "20", "category": "Mathematics"},
    {"question": "What is 2 to the power of 10?", "expected": "1024", "category": "Mathematics"},

    # Science
    {"question": "What is the chemical formula for water?", "expected": "H2O", "category": "Science"},
    {"question": "Which planet is known as the Red Planet?", "expected": "Mars", "category": "Science"},
    {"question": "What gas do plants absorb during photosynthesis?", "expected": "Carbon dioxide", "category": "Science"},
    {"question": "What is the chemical symbol for gold?", "expected": "Au", "category": "Science"},
    {"question": "What force keeps planets in orbit around the Sun?", "expected": "Gravity", "category": "Science"},
    {"question": "What is the closest star to Earth?", "expected": "The Sun", "category": "Science"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "expected": "100", "category": "Science"},
    {"question": "Which organ pumps blood through the human body?", "expected": "Heart", "category": "Science"},
    {"question": "What is the approximate speed of light in a vacuum in kilometers per second?", "expected": "300000", "category": "Science"},
    {"question": "What is the chemical symbol for oxygen?", "expected": "O", "category": "Science"},

    # Technology
    {"question": "What does CPU stand for?", "expected": "Central Processing Unit", "category": "Technology"},
    {"question": "What does RAM stand for?", "expected": "Random Access Memory", "category": "Technology"},
    {"question": "What does HTML stand for?", "expected": "HyperText Markup Language", "category": "Technology"},
    {"question": "Which programming language is commonly associated with the .py file extension?", "expected": "Python", "category": "Technology"},
    {"question": "What does URL stand for?", "expected": "Uniform Resource Locator", "category": "Technology"},
    {"question": "What does SQL stand for?", "expected": "Structured Query Language", "category": "Technology"},
    {"question": "Which data structure follows the Last In, First Out principle?", "expected": "Stack", "category": "Technology"},
    {"question": "What is the default port number for HTTP?", "expected": "80", "category": "Technology"},
    {"question": "What does AI stand for?", "expected": "Artificial Intelligence", "category": "Technology"},
    {"question": "Which protocol is commonly used to securely browse websites?", "expected": "HTTPS", "category": "Technology"},

    # General knowledge
    {"question": "How many days are there in a leap year?", "expected": "366", "category": "General Knowledge"},
    {"question": "How many sides does a hexagon have?", "expected": "6", "category": "General Knowledge"},
    {"question": "What is the largest mammal in the world?", "expected": "Blue whale", "category": "General Knowledge"},
    {"question": "How many colors are traditionally listed in a rainbow?", "expected": "7", "category": "General Knowledge"},
    {"question": "Which language is primarily spoken in Brazil?", "expected": "Portuguese", "category": "General Knowledge"},
    {"question": "What is the currency of Japan?", "expected": "Yen", "category": "General Knowledge"},
    {"question": "How many players from one team are on the field in a standard soccer match?", "expected": "11", "category": "General Knowledge"},
    {"question": "Which instrument is used to measure temperature?", "expected": "Thermometer", "category": "General Knowledge"},
    {"question": "What is the freezing point of water in degrees Celsius?", "expected": "0", "category": "General Knowledge"},
    {"question": "Which natural satellite orbits Earth?", "expected": "Moon", "category": "General Knowledge"},
]


# =========================================================
# 3. PROMPT VARIATION GENERATOR
# =========================================================

def generate_prompt_variants(prompt):
    prompt = prompt.strip()

    words = prompt.split()

    # Add a deterministic spelling perturbation where possible.
    typo_prompt = prompt

    for word in words:
        if len(word) >= 5 and word.isalpha():
            typo = word[:-1] + word[-1] * 2
            typo_prompt = prompt.replace(word, typo, 1)
            break

    if typo_prompt == prompt:
        typo_prompt = prompt + " Please answer accuratley."

    return {
        "Original": prompt,
        "Instruction Added":
            "Please answer the following question clearly: " + prompt,
        "Direct Instruction":
            "Provide a direct and accurate answer to this question: " + prompt,
        "Detailed Instruction":
            "Read the question carefully and explain your answer step by step: "
            + prompt,
        "Concise Instruction":
            "Answer briefly and precisely: " + prompt,
        "Typos Introduced": typo_prompt
    }


# =========================================================
# 4. MODEL INFERENCE
# =========================================================

def ask_model(prompt, timeout=180):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42
            }
        },
        timeout=timeout
    )

    response.raise_for_status()

    result = response.json()
    answer = result.get("response", "").strip()

    if not answer:
        raise ValueError("The model returned an empty response.")

    return answer


# =========================================================
# 5. ANSWER EVALUATION
# =========================================================

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def is_correct(response, expected):
    """
    Basic reference matching for short factual answers.
    This is not suitable for every task type.
    """

    answer = normalize(response)
    target = normalize(expected)

    if not answer or not target:
        return False

    # Exact match.
    if answer == target:
        return True

    # Short reference answers can be matched as complete tokens.
    if len(target.split()) == 1:
        return bool(
            re.search(
                r"\b" + re.escape(target) + r"\b",
                answer
            )
        )

    # Multiword reference answer.
    return target in answer


# =========================================================
# 6. EXPERIMENT HISTORY
# =========================================================

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_FILE)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def save_history(results):
    if results.empty:
        return

    # Append new records without deleting previous experiments.
    file_exists = (
        os.path.exists(HISTORY_FILE)
        and os.path.getsize(HISTORY_FILE) > 0
    )

    results.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )


# =========================================================
# 7. RESPONSE CONSISTENCY
# =========================================================

def calculate_response_similarity(answers):
    """
    Calculate average pairwise TF-IDF cosine similarity.
    This measures lexical overlap, not factual correctness.
    """

    answers = [str(answer).strip() for answer in answers]

    if len(answers) < 2:
        return None

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(answers)
        similarity_matrix = cosine_similarity(matrix)

        pair_scores = []

        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                pair_scores.append(similarity_matrix[i, j])

        if not pair_scores:
            return None

        return float(sum(pair_scores) / len(pair_scores))

    except ValueError:
        # For example, all responses may contain no usable words.
        if len(set(answers)) == 1:
            return 1.0

        return 0.0


# =========================================================
# 8. APPLICATION HEADER
# =========================================================

st.title("🧪 Prompt Sensitivity Lab")

st.subheader(
    "Language Model Evaluation With Prompt Sensitivity Quantified"
)

st.markdown(
    """
    Evaluate how prompt wording affects language-model responses
    using automated inference, benchmark accuracy, response similarity,
    sensitivity metrics and persistent experiment history.
    """
)

st.sidebar.header("Experiment Configuration")

st.sidebar.write(f"**Model:** `{MODEL_NAME}`")
st.sidebar.write("**Inference:** Local Ollama")

question_count = st.sidebar.selectbox(
    "Number of benchmark questions",
    [3, 5, 10],
    index=0
)

timeout = st.sidebar.slider(
    "Timeout per model request (seconds)",
    min_value=30,
    max_value=300,
    value=180,
    step=30
)

st.sidebar.caption(
    "Each question is tested using six prompt variants."
)

st.sidebar.markdown("---")

if st.sidebar.button("Check Ollama Connection"):
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        response.raise_for_status()

        available_models = [
            model.get("name", "")
            for model in response.json().get("models", [])
        ]

        st.sidebar.success("Ollama is responding.")

        if any(
            name == MODEL_NAME or name.startswith(MODEL_NAME + ":")
            for name in available_models
        ):
            st.sidebar.success("Required model is installed.")
        else:
            st.sidebar.warning(
                f"Model not found. Run: ollama pull {MODEL_NAME}"
            )

    except requests.RequestException:
        st.sidebar.error(
            "Ollama is not reachable. Start Ollama and try again."
        )


# =========================================================
# 9. SINGLE PROMPT EXPERIMENT
# =========================================================

st.header("1. Single Prompt Experiment")

with st.expander("Test your own prompt", expanded=False):

    custom_prompt = st.text_area(
        "Enter a question",
        value="What is the capital of Japan?",
        key="custom_prompt"
    )

    custom_expected = st.text_input(
        "Expected answer",
        value="Tokyo",
        key="custom_expected"
    )

    if st.button("Run Custom Prompt Experiment"):

        if not custom_prompt.strip():
            st.error("Please enter a prompt.")

        elif not custom_expected.strip():
            st.error("Please enter an expected answer.")

        else:
            custom_variants = generate_prompt_variants(
                custom_prompt
            )

            custom_rows = []
            progress = st.progress(0)
            status = st.empty()

            try:
                for index, (name, variant) in enumerate(
                    custom_variants.items()
                ):

                    status.write(
                        f"Generating response: {name}"
                    )

                    answer = ask_model(variant, timeout)

                    custom_rows.append({
                        "Experiment_ID": str(uuid.uuid4())[:8],
                        "Timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "Question": custom_prompt,
                        "Category": "Custom",
                        "Variant": name,
                        "Prompt": variant,
                        "Expected_Answer": custom_expected,
                        "Model_Response": answer,
                        "Correct": is_correct(
                            answer, custom_expected
                        ),
                        "Score": int(
                            is_correct(answer, custom_expected)
                        )
                    })

                    progress.progress(
                        (index + 1) / len(custom_variants)
                    )

                custom_df = pd.DataFrame(custom_rows)

                st.session_state["custom_results"] = custom_df

                save_history(custom_df)

                status.success(
                    "Custom experiment completed and saved."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to Ollama. Make sure Ollama is running."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The model timed out. Try increasing the timeout."
                )

            except (
                requests.exceptions.RequestException,
                ValueError
            ) as error:
                st.error(f"Experiment failed: {error}")


if "custom_results" in st.session_state:

    custom_df = st.session_state["custom_results"]

    st.subheader("Custom Experiment Results")

    st.dataframe(
        custom_df[
            ["Variant", "Model_Response", "Expected_Answer", "Correct"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Custom Experiment CSV",
        data=custom_df.to_csv(index=False).encode("utf-8"),
        file_name="custom_prompt_experiment.csv",
        mime="text/csv",
        key="download_custom"
    )


# =========================================================
# 10. BENCHMARK EXPERIMENT
# =========================================================

st.markdown("---")
st.header("2. Automated Benchmark Evaluation")

dataset = BENCHMARK_QUESTIONS[:question_count]

with st.expander("View benchmark dataset"):

    st.dataframe(
        pd.DataFrame(dataset),
        use_container_width=True,
        hide_index=True
    )

st.write(
    f"The benchmark contains {len(dataset)} questions. "
    f"Each question will be tested using six prompt variations."
)

st.caption(
    f"Total model requests per run: {len(dataset) * 6}"
)

if st.button("Run Complete Benchmark", type="primary"):

    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat(timespec="seconds")

    rows = []
    total = len(dataset) * 6
    completed = 0

    progress = st.progress(0)
    status = st.empty()

    try:
        for case_number, case in enumerate(dataset, start=1):

            variants = generate_prompt_variants(
                case["question"]
            )

            for variant_name, variant_prompt in variants.items():

                status.write(
                    f"Question {case_number}/{len(dataset)} | "
                    f"Variant: {variant_name}"
                )

                answer = ask_model(
                    variant_prompt,
                    timeout
                )

                correct = is_correct(
                    answer,
                    case["expected"]
                )

                rows.append({
                    "Experiment_ID": experiment_id,
                    "Timestamp": timestamp,
                    "Question": case["question"],
                    "Category": case["category"],
                    "Variant": variant_name,
                    "Prompt": variant_prompt,
                    "Expected_Answer": case["expected"],
                    "Model_Response": answer,
                    "Correct": correct,
                    "Score": int(correct)
                })

                completed += 1
                progress.progress(completed / total)

        benchmark_df = pd.DataFrame(rows)

        # Store current run in Streamlit session.
        st.session_state["benchmark_results"] = benchmark_df

        # Append this run to persistent CSV history.
        save_history(benchmark_df)

        status.success(
            f"Benchmark completed. Experiment ID: {experiment_id}"
        )

    except requests.exceptions.ConnectionError:
        status.error(
            "Cannot connect to Ollama. Open Ollama and retry."
        )

    except requests.exceptions.Timeout:
        status.error(
            "The model timed out. Retry with fewer questions "
            "or increase the timeout."
        )

    except (
        requests.exceptions.RequestException,
        ValueError
    ) as error:
        status.error(f"Benchmark failed: {error}")


# =========================================================
# 11. CURRENT RUN RESULTS
# =========================================================

if "benchmark_results" in st.session_state:

    df = st.session_state["benchmark_results"]

    st.markdown("---")
    st.header("3. Detailed Evaluation Results")

    st.dataframe(
        df[
            [
                "Question",
                "Category",
                "Variant",
                "Model_Response",
                "Expected_Answer",
                "Correct"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.header("4. Performance Metrics")

    overall_accuracy = df["Score"].mean()

    pivot = df.pivot(
        index="Question",
        columns="Variant",
        values="Score"
    )

    variant_columns = [
        column
        for column in pivot.columns
        if column != "Original"
    ]

    sensitivity = None

    if "Original" in pivot.columns and variant_columns:

        original_scores = pivot["Original"]

        differences = pivot[variant_columns].sub(
            original_scores,
            axis=0
        ).abs()

        if not differences.empty:
            sensitivity = float(
                differences.to_numpy().mean()
            )

    response_similarity = []

    for question, group in df.groupby("Question"):

        similarity = calculate_response_similarity(
            group["Model_Response"].tolist()
        )

        if similarity is not None:
            response_similarity.append({
                "Question": question,
                "Mean Response Similarity": similarity
            })

    average_similarity = (
        sum(
            row["Mean Response Similarity"]
            for row in response_similarity
        ) / len(response_similarity)
        if response_similarity else None
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Answer-Match Rate",
        f"{overall_accuracy:.1%}"
    )

    col2.metric(
        "Mean Absolute Sensitivity",
        f"{sensitivity:.3f}"
        if sensitivity is not None else "N/A"
    )

    col3.metric(
        "Response Similarity",
        f"{average_similarity:.3f}"
        if average_similarity is not None else "N/A"
    )

    st.caption(
        "Sensitivity is the mean absolute change in binary answer-match "
        "scores between the original prompt and each variation. "
        "Response similarity uses TF-IDF cosine similarity. "
        "Neither metric captures every aspect of answer quality."
    )

    # Accuracy chart.
    st.header("5. Accuracy by Prompt Variation")

    accuracy_by_variant = (
        df.groupby("Variant", sort=False)["Score"]
        .mean()
        .rename("Accuracy")
    )

    st.bar_chart(accuracy_by_variant)

    # Category comparison.
    st.header("6. Category-Level Analysis")

    category_table = (
        df.groupby(["Category", "Variant"])["Score"]
        .mean()
        .unstack()
    )

    st.dataframe(
        category_table,
        use_container_width=True
    )

    # Response consistency details.
    st.header("7. Response Consistency")

    if response_similarity:

        similarity_df = pd.DataFrame(response_similarity)

        st.dataframe(
            similarity_df,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots()

        ax.bar(
            similarity_df["Question"],
            similarity_df["Mean Response Similarity"]
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean TF-IDF cosine similarity")
        ax.set_xlabel("Question")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.header("8. Download Current Results")

    st.download_button(
        "Download Benchmark CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prompt_sensitivity_benchmark.csv",
        mime="text/csv",
        key="download_benchmark"
    )


# =========================================================
# 12. PERSISTENT EXPERIMENT HISTORY
# =========================================================

st.markdown("---")
st.header("9. Experiment History")

history = load_history()

if history.empty:

    st.info(
        "No saved experiments yet. Run a custom experiment "
        "or the benchmark to create history."
    )

else:

    st.write(
        f"Saved experiment records: {len(history)}"
    )

    if "Experiment_ID" in history.columns:

        experiment_ids = (
            history["Experiment_ID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_experiment = st.selectbox(
            "Filter by experiment ID",
            ["All Experiments"] + experiment_ids
        )

        if selected_experiment != "All Experiments":

            filtered_history = history[
                history["Experiment_ID"].astype(str)
                == selected_experiment
            ]

        else:
            filtered_history = history

    else:
        filtered_history = history

    st.dataframe(
        filtered_history,
        use_container_width=True,
        hide_index=True
    )

    if "Correct" in filtered_history.columns:

        history_accuracy = pd.to_numeric(
            filtered_history["Correct"],
            errors="coerce"
        ).mean()

        if pd.notna(history_accuracy):

            st.metric(
                "Historical Answer-Match Rate",
                f"{history_accuracy:.1%}"
            )

    st.download_button(
        "Download All Experiment History",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="all_experiment_history.csv",
        mime="text/csv",
        key="download_history"
    )

st.markdown("---")

st.caption(
    "Prompt Sensitivity Lab | Local model inference using Ollama | "
    "Experimental prototype"
)
