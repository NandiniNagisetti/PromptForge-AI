import os
import re
import uuid
import time
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
    page_title="PromptForge AI",
    page_icon="🧪",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_NAME = "qwen2.5:3b"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(
    BASE_DIR, "experiment_history_enhanced.csv"
)

VARIANT_ORDER = [
    "Original",
    "Instruction Added",
    "Direct Instruction",
    "Detailed Instruction",
    "Concise Instruction",
    "Typos Introduced"
]


# =========================================================
# 2. BENCHMARK DATASET - 50 QUESTIONS
# =========================================================

BENCHMARK_QUESTIONS = [
    # Geography
    {"question": "What is the capital of France?",
     "expected": "Paris", "category": "Geography"},
    {"question": "What is the capital of Japan?",
     "expected": "Tokyo", "category": "Geography"},
    {"question": "What is the capital of Australia?",
     "expected": "Canberra", "category": "Geography"},
    {"question": "What is the capital of Italy?",
     "expected": "Rome", "category": "Geography"},
    {"question": "What is the capital of Canada?",
     "expected": "Ottawa", "category": "Geography"},
    {"question": "What is the largest ocean on Earth?",
     "expected": "Pacific Ocean", "category": "Geography"},
    {"question": "Which is the largest continent by area?",
     "expected": "Asia", "category": "Geography"},
    {"question": "Which country is home to the Great Pyramid of Giza?",
     "expected": "Egypt", "category": "Geography"},
    {"question": "What is the capital of India?",
     "expected": "New Delhi", "category": "Geography"},
    {"question": "Which river flows through London?",
     "expected": "Thames", "category": "Geography"},

    # Mathematics
    {"question": "What is 12 multiplied by 8?",
     "expected": "96", "category": "Mathematics"},
    {"question": "What is the square of 15?",
     "expected": "225", "category": "Mathematics"},
    {"question": "What is the square root of 144?",
     "expected": "12", "category": "Mathematics"},
    {"question": "What is 125 plus 275?",
     "expected": "400", "category": "Mathematics"},
    {"question": "What is 1000 minus 365?",
     "expected": "635", "category": "Mathematics"},
    {"question": "What is 20 percent of 250?",
     "expected": "50", "category": "Mathematics"},
    {"question": "How many degrees are in a right angle?",
     "expected": "90", "category": "Mathematics"},
    {"question": "What is 7 factorial?",
     "expected": "5040", "category": "Mathematics"},
    {"question": "What is the perimeter of a square with side length 5?",
     "expected": "20", "category": "Mathematics"},
    {"question": "What is 2 to the power of 10?",
     "expected": "1024", "category": "Mathematics"},

    # Science
    {"question": "What is the chemical formula for water?",
     "expected": "H2O", "category": "Science"},
    {"question": "Which planet is known as the Red Planet?",
     "expected": "Mars", "category": "Science"},
    {"question": "What gas do plants absorb during photosynthesis?",
     "expected": "Carbon dioxide", "category": "Science"},
    {"question": "What is the chemical symbol for gold?",
     "expected": "Au", "category": "Science"},
    {"question": "What force keeps planets in orbit around the Sun?",
     "expected": "Gravity", "category": "Science"},
    {"question": "What is the closest star to Earth?",
     "expected": "The Sun", "category": "Science"},
    {"question": "What is the boiling point of water at sea level in Celsius?",
     "expected": "100", "category": "Science"},
    {"question": "Which organ pumps blood through the human body?",
     "expected": "Heart", "category": "Science"},
    {"question": "What is the approximate speed of light in a vacuum in kilometers per second?",
     "expected": "300000", "category": "Science"},
    {"question": "What is the chemical symbol for oxygen?",
     "expected": "O", "category": "Science"},

    # Technology
    {"question": "What does CPU stand for?",
     "expected": "Central Processing Unit", "category": "Technology"},
    {"question": "What does RAM stand for?",
     "expected": "Random Access Memory", "category": "Technology"},
    {"question": "What does HTML stand for?",
     "expected": "HyperText Markup Language", "category": "Technology"},
    {"question": "Which programming language is commonly associated with the .py file extension?",
     "expected": "Python", "category": "Technology"},
    {"question": "What does URL stand for?",
     "expected": "Uniform Resource Locator", "category": "Technology"},
    {"question": "What does SQL stand for?",
     "expected": "Structured Query Language", "category": "Technology"},
    {"question": "Which data structure follows the Last In, First Out principle?",
     "expected": "Stack", "category": "Technology"},
    {"question": "What is the default port number for HTTP?",
     "expected": "80", "category": "Technology"},
    {"question": "What does AI stand for?",
     "expected": "Artificial Intelligence", "category": "Technology"},
    {"question": "Which protocol is commonly used to securely browse websites?",
     "expected": "HTTPS", "category": "Technology"},

    # General Knowledge
    {"question": "How many days are there in a leap year?",
     "expected": "366", "category": "General Knowledge"},
    {"question": "How many sides does a hexagon have?",
     "expected": "6", "category": "General Knowledge"},
    {"question": "What is the largest mammal in the world?",
     "expected": "Blue whale", "category": "General Knowledge"},
    {"question": "How many colors are traditionally listed in a rainbow?",
     "expected": "7", "category": "General Knowledge"},
    {"question": "Which language is primarily spoken in Brazil?",
     "expected": "Portuguese", "category": "General Knowledge"},
    {"question": "What is the currency of Japan?",
     "expected": "Yen", "category": "General Knowledge"},
    {"question": "How many players from one team are on the field in a standard soccer match?",
     "expected": "11", "category": "General Knowledge"},
    {"question": "Which instrument is used to measure temperature?",
     "expected": "Thermometer", "category": "General Knowledge"},
    {"question": "What is the freezing point of water in degrees Celsius?",
     "expected": "0", "category": "General Knowledge"},
    {"question": "Which natural satellite orbits Earth?",
     "expected": "Moon", "category": "General Knowledge"}
]


# =========================================================
# 3. PROMPT VARIATION GENERATOR
# =========================================================

def generate_prompt_variants(prompt):
    prompt = prompt.strip()
    words = prompt.split()
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
    """Send exactly the supplied prompt to the local Ollama model."""

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
    """Basic reference matching; not a complete factuality evaluator."""

    answer = normalize(response)
    target = normalize(expected)

    if not answer or not target:
        return False

    if answer == target:
        return True

    # Match short answers as whole words, not substrings.
    if len(target.split()) == 1:
        return bool(
            re.search(r"\b" + re.escape(target) + r"\b", answer)
        )

    return target in answer


# =========================================================
# 6. EXPERIMENT HISTORY
# =========================================================

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(HISTORY_FILE)
    except (
        pd.errors.EmptyDataError,
        OSError,
        ValueError
    ):
        return pd.DataFrame()


def save_history(results):
    """Append results without deleting earlier experiments."""

    if results is None or results.empty:
        return

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
# 7. RESPONSE SIMILARITY
# =========================================================

def calculate_response_similarity(answers):
    """Average pairwise TF-IDF cosine similarity."""

    answers = [
        str(answer).strip()
        for answer in answers
        if pd.notna(answer) and str(answer).strip()
    ]

    if len(answers) < 2:
        return None

    try:
        matrix = TfidfVectorizer().fit_transform(answers)
        similarity_matrix = cosine_similarity(matrix)

        pair_scores = [
            similarity_matrix[i, j]
            for i in range(len(answers))
            for j in range(i + 1, len(answers))
        ]

        return (
            float(sum(pair_scores) / len(pair_scores))
            if pair_scores else None
        )

    except ValueError:
        # TF-IDF may fail when all texts contain no usable tokens.
        return 1.0 if len(set(answers)) == 1 else 0.0


# =========================================================
# 8. APPLICATION HEADER AND SIDEBAR
# =========================================================

st.title("⚡ PromptForge AI")
st.subheader(
    "Intelligent Prompt Engineering, Optimization and LLM Evaluation Platform"
)

st.markdown(
    """
    Evaluate how prompt wording affects language-model responses using
    local inference, benchmark answer matching, response similarity,
    sensitivity metrics, visualizations, and persistent experiment history.
    """
)

st.sidebar.header("Experiment Configuration")
st.sidebar.write(f"**Model:** `{MODEL_NAME}`")
st.sidebar.write("**Inference:** Local Ollama")

question_count = st.sidebar.selectbox(
    "Number of benchmark questions",
    [3, 5, 10, 20, 30, 40, 50],
    index=0
)

timeout = st.sidebar.slider(
    "Timeout per model request (seconds)",
    min_value=30,
    max_value=300,
    value=180,
    step=30
)

repeat_trials = st.sidebar.selectbox(
    "Repeated trials per prompt variant",
    [1, 2, 3],
    index=0,
    help="More trials improve the amount of observed data but increase runtime."
)

st.sidebar.caption(
    "Each question is tested using six prompt variants."
)

st.sidebar.markdown("---")

if st.sidebar.button("Check Ollama Connection"):
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
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

    except requests.RequestException as error:
        st.sidebar.error(
            f"Ollama is not reachable: {error}"
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
        # Remove old results so a failed run cannot display stale results.
        st.session_state.pop("custom_results", None)

        if not custom_prompt.strip():
            st.error("Please enter a prompt.")

        elif not custom_expected.strip():
            st.error("Please enter an expected answer.")

        else:
            variants = generate_prompt_variants(custom_prompt)
            rows = []
            experiment_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().isoformat(timespec="seconds")

            progress = st.progress(0)
            status = st.empty()

            try:
                for index, (name, variant) in enumerate(variants.items()):
                    status.write(f"Generating response: {name}")

                    started_at = time.perf_counter()
                    answer = ask_model(variant, timeout)
                    response_seconds = time.perf_counter() - started_at
                    correct = is_correct(answer, custom_expected)

                    rows.append({
                        "Experiment_ID": experiment_id,
                        "Timestamp": timestamp,
                        "Question": custom_prompt,
                        "Category": "Custom",
                        "Variant": name,
                        "Prompt": variant,
                        "Expected_Answer": custom_expected,
                        "Model_Response": answer,
                        "Response_Time_Seconds": round(response_seconds, 3),
                        "Trial": 1,
                        "Correct": correct,
                        "Score": int(correct)
                    })

                    progress.progress((index + 1) / len(variants))

                custom_df = pd.DataFrame(rows)
                save_history(custom_df)
                st.session_state["custom_results"] = custom_df
                status.success(
                    f"Custom experiment completed. ID: {experiment_id}"
                )

            except requests.exceptions.ConnectionError:
                status.error(
                    "Cannot connect to Ollama. Start Ollama and retry."
                )

            except requests.exceptions.Timeout:
                status.error(
                    "The model request timed out. Increase the timeout "
                    "or try again."
                )

            except (
                requests.exceptions.RequestException,
                ValueError
            ) as error:
                status.error(f"Experiment failed: {error}")

            except Exception as error:
                status.error(f"Unexpected error: {error}")


if "custom_results" in st.session_state:
    custom_df = st.session_state["custom_results"]

    st.subheader("Custom Experiment Results")
    st.caption(
        "The Prompt column shows the exact text sent to Ollama."
    )

    st.dataframe(
        custom_df[
            [
                "Variant", "Prompt", "Model_Response",
                "Expected_Answer", "Correct"
            ]
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
# 10. AUTOMATED BENCHMARK
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
    "Each question is tested using six prompt variations."
)

total_requests = len(dataset) * len(VARIANT_ORDER) * repeat_trials
st.caption(f"Total model requests per run: {total_requests}")

if st.button("Run Complete Benchmark", type="primary"):
    # Clear old benchmark results before attempting a new run.
    st.session_state.pop("benchmark_results", None)

    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = []

    total = len(dataset) * len(VARIANT_ORDER) * repeat_trials
    completed = 0

    progress = st.progress(0)
    status = st.empty()

    try:
        for case_number, case in enumerate(dataset, start=1):
            variants = generate_prompt_variants(case["question"])

            for variant_name, variant_prompt in variants.items():
                for trial_number in range(1, repeat_trials + 1):
                    status.write(
                        f"Question {case_number}/{len(dataset)} | "
                        f"{case['category']} | {variant_name} | "
                        f"Trial {trial_number}/{repeat_trials}"
                    )

                    started_at = time.perf_counter()
                    answer = ask_model(variant_prompt, timeout)
                    response_seconds = time.perf_counter() - started_at
                    correct = is_correct(answer, case["expected"])

                    rows.append({
                        "Experiment_ID": experiment_id,
                        "Timestamp": timestamp,
                        "Question": case["question"],
                        "Category": case["category"],
                        "Variant": variant_name,
                        "Trial": trial_number,
                        "Prompt": variant_prompt,
                        "Expected_Answer": case["expected"],
                        "Model_Response": answer,
                        "Response_Time_Seconds": round(response_seconds, 3),
                        "Correct": correct,
                        "Score": int(correct)
                    })

                    completed += 1
                    progress.progress(completed / total)

        benchmark_df = pd.DataFrame(rows)

        # Store and save only after every requested inference succeeds.
        save_history(benchmark_df)
        st.session_state["benchmark_results"] = benchmark_df

        status.success(
            f"Benchmark completed. Experiment ID: {experiment_id}"
        )

    except requests.exceptions.ConnectionError:
        status.error(
            "Cannot connect to Ollama. Start Ollama and retry."
        )

    except requests.exceptions.Timeout:
        status.error(
            "A model request timed out. Try fewer questions or increase "
            "the timeout."
        )

    except (
        requests.exceptions.RequestException,
        ValueError
    ) as error:
        status.error(f"Benchmark failed: {error}")

    except Exception as error:
        status.error(f"Unexpected error: {error}")


# =========================================================
# 11. RESULTS AND METRICS
# =========================================================

if "benchmark_results" in st.session_state:
    df = st.session_state["benchmark_results"].copy()

    st.markdown("---")
    st.header("3. Detailed Evaluation Results")

    categories_available = (
        ["All categories"]
        + sorted(df["Category"].dropna().unique().tolist())
    )

    selected_category = st.selectbox(
        "Filter detailed results by category",
        categories_available,
        key="results_category_filter"
    )

    display_df = (
        df if selected_category == "All categories"
        else df[df["Category"] == selected_category]
    )

    st.dataframe(
        display_df[
            [
                "Question", "Category", "Variant", "Trial",
                "Model_Response", "Expected_Answer", "Correct",
                "Response_Time_Seconds"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.header("4. Performance Metrics")

    overall_accuracy = pd.to_numeric(
        df["Score"], errors="coerce"
    ).mean()

    # Average repeated trials per question and prompt variant.
    pivot = df.pivot_table(
        index="Question",
        columns="Variant",
        values="Score",
        aggfunc="mean"
    )

    variant_columns = [
        col for col in pivot.columns if col != "Original"
    ]

    sensitivity = None
    differences = None

    if "Original" in pivot.columns and variant_columns:
        differences = pivot[variant_columns].sub(
            pivot["Original"], axis=0
        ).abs()

        if not differences.empty:
            sensitivity = float(differences.to_numpy().mean())

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
        sum(x["Mean Response Similarity"] for x in response_similarity)
        / len(response_similarity)
        if response_similarity else None
    )

    mean_response_time = pd.to_numeric(
        df["Response_Time_Seconds"], errors="coerce"
    ).mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Answer-Match Rate",
        f"{overall_accuracy:.1%}"
        if pd.notna(overall_accuracy) else "N/A"
    )

    col2.metric(
        "Mean Absolute Sensitivity",
        f"{sensitivity:.3f}" if sensitivity is not None else "N/A"
    )

    col3.metric(
        "Response Similarity",
        f"{average_similarity:.3f}"
        if average_similarity is not None else "N/A"
    )

    col4.metric(
        "Mean Response Time",
        f"{mean_response_time:.2f} s"
        if pd.notna(mean_response_time) else "N/A"
    )

    st.caption(
        "Sensitivity is the mean absolute change in binary answer-match "
        "scores between the original prompt and its variations. Response "
        "similarity measures lexical overlap using TF-IDF cosine similarity. "
        "Neither metric fully measures answer quality or factuality."
    )


    # =====================================================
    # 12. ADVANCED VISUALIZATIONS
    # =====================================================

    st.header("5. Advanced Prompt Sensitivity Analysis")

    variant_order = [
        variant for variant in VARIANT_ORDER
        if variant in df["Variant"].unique()
    ]

    plot_df = df.copy()
    plot_df["Score"] = pd.to_numeric(
        plot_df["Score"], errors="coerce"
    )
    plot_df["Response_Time_Seconds"] = pd.to_numeric(
        plot_df["Response_Time_Seconds"], errors="coerce"
    )

    plot_df = plot_df.dropna(
        subset=["Variant", "Category", "Question", "Score"]
    )

    if plot_df.empty:
        st.warning("No valid benchmark results are available to plot.")

    else:
        # -------------------------------------------------
        # 5.1 Accuracy with Wilson confidence intervals
        # -------------------------------------------------

        st.subheader("5.1 Prompt Accuracy with 95% Confidence Intervals")

        accuracy_rows = []

        for variant in variant_order:
            values = plot_df.loc[
                plot_df["Variant"] == variant, "Score"
            ].dropna()

            n = len(values)

            if n == 0:
                continue

            p = values.mean()
            z = 1.96
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator

            margin = (
                z * (
                    p * (1 - p) / n + z**2 / (4 * n**2)
                ) ** 0.5 / denominator
            )

            accuracy_rows.append({
                "Variant": variant,
                "Accuracy": p,
                "Lower": max(0, center - margin),
                "Upper": min(1, center + margin),
                "Trials": n
            })

        accuracy_df = pd.DataFrame(accuracy_rows)

        if not accuracy_df.empty:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            positions = list(range(len(accuracy_df)))

            ax.barh(
                positions,
                accuracy_df["Accuracy"] * 100,
                height=0.62
            )

            lower_error = (
                accuracy_df["Accuracy"] - accuracy_df["Lower"]
            ) * 100
            upper_error = (
                accuracy_df["Upper"] - accuracy_df["Accuracy"]
            ) * 100

            ax.errorbar(
                accuracy_df["Accuracy"] * 100,
                positions,
                xerr=[lower_error, upper_error],
                fmt="none",
                ecolor="black",
                capsize=4
            )

            ax.set_yticks(positions)
            ax.set_yticklabels(accuracy_df["Variant"])
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_xlabel("Answer-match rate (%)")
            ax.set_title("Prompt Variant Performance")
            ax.grid(axis="x", alpha=0.25)

            for y, value in enumerate(accuracy_df["Accuracy"]):
                ax.text(value * 100 + 1, y, f"{value:.1%}", va="center")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Intervals use the 95% Wilson method and describe "
                "uncertainty in the observed answer-match proportion."
            )


        # -------------------------------------------------
        # 5.2 Category-by-variant heatmap
        # -------------------------------------------------

        st.subheader("5.2 Category × Prompt Accuracy Heatmap")

        category_table = (
            plot_df.groupby(["Category", "Variant"])["Score"]
            .mean()
            .unstack()
            .reindex(columns=variant_order)
        )

        if not category_table.empty:
            fig, ax = plt.subplots(
                figsize=(12, max(4, 0.8 * len(category_table)))
            )

            heat = ax.imshow(
                category_table.to_numpy(dtype=float),
                aspect="auto",
                vmin=0,
                vmax=1
            )

            ax.set_xticks(range(len(category_table.columns)))
            ax.set_xticklabels(
                category_table.columns,
                rotation=35,
                ha="right"
            )
            ax.set_yticks(range(len(category_table.index)))
            ax.set_yticklabels(category_table.index)
            ax.set_xlabel("Prompt variation")
            ax.set_ylabel("Benchmark category")
            ax.set_title("Answer-Match Rate by Category and Prompt")

            for row in range(category_table.shape[0]):
                for col in range(category_table.shape[1]):
                    value = category_table.iloc[row, col]

                    if pd.notna(value):
                        ax.text(
                            col, row, f"{value:.0%}",
                            ha="center", va="center"
                        )

            fig.colorbar(heat, ax=ax, label="Accuracy")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


        # -------------------------------------------------
        # 5.3 Accuracy versus response time
        # -------------------------------------------------

        st.subheader("5.3 Accuracy vs. Response Time")

        tradeoff = (
            plot_df.groupby("Variant")
            .agg(
                Accuracy=("Score", "mean"),
                Mean_Latency=("Response_Time_Seconds", "mean"),
                Number_of_Responses=("Score", "count")
            )
            .reindex(variant_order)
            .dropna(subset=["Accuracy", "Mean_Latency"])
        )

        if not tradeoff.empty:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(
                tradeoff["Mean_Latency"],
                tradeoff["Accuracy"] * 100,
                s=100
            )

            for variant, row in tradeoff.iterrows():
                ax.annotate(
                    variant,
                    (row["Mean_Latency"], row["Accuracy"] * 100),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8
                )

            ax.set_xlabel("Mean response time (seconds)")
            ax.set_ylabel("Answer-match rate (%)")
            ax.set_ylim(-5, 105)
            ax.set_title("Accuracy–Latency Trade-off")
            ax.grid(alpha=0.25)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            tradeoff_table = tradeoff.reset_index().rename(
                columns={
                    "Variant": "Variant",
                    "Accuracy": "Accuracy (0–1)",
                    "Mean_Latency": "Mean latency (seconds)",
                    "Number_of_Responses": "Responses"
                }
            )

            st.dataframe(
                tradeoff_table,
                use_container_width=True,
                hide_index=True
            )


        # -------------------------------------------------
        # 5.4 Question-level sensitivity heatmap
        # -------------------------------------------------

        st.subheader("5.4 Question-Level Sensitivity Map")

        score_matrix = plot_df.pivot_table(
            index="Question",
            columns="Variant",
            values="Score",
            aggfunc="mean"
        ).reindex(columns=variant_order)

        if "Original" in score_matrix.columns:
            comparison_columns = [
                col for col in score_matrix.columns
                if col != "Original"
            ]

            if comparison_columns:
                sensitivity_matrix = score_matrix[
                    comparison_columns
                ].sub(score_matrix["Original"], axis=0).abs()

                question_sensitivity = (
                    sensitivity_matrix.mean(axis=1, skipna=True)
                    .sort_values(ascending=False)
                )

                top_questions = question_sensitivity.head(20).index
                display_matrix = sensitivity_matrix.loc[top_questions]

                fig, ax = plt.subplots(
                    figsize=(12, max(5, 0.38 * len(display_matrix)))
                )

                heat = ax.imshow(
                    display_matrix.to_numpy(dtype=float),
                    aspect="auto",
                    vmin=0,
                    vmax=1
                )

                ax.set_xticks(range(len(display_matrix.columns)))
                ax.set_xticklabels(
                    display_matrix.columns,
                    rotation=35,
                    ha="right"
                )
                ax.set_yticks(range(len(display_matrix.index)))
                ax.set_yticklabels(display_matrix.index, fontsize=8)
                ax.set_xlabel("Prompt variation compared with Original")
                ax.set_ylabel("Benchmark question")
                ax.set_title("Absolute Change in Answer-Match Score")

                fig.colorbar(
                    heat, ax=ax,
                    label="Absolute score change (0–1)"
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Questions are ranked by mean absolute score change. "
                    "A value of 1 means the binary answer-match score changed "
                    "completely; 0 means it did not change."
                )

                st.dataframe(
                    question_sensitivity.rename(
                        "Mean Absolute Sensitivity"
                    ).reset_index(),
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "At least one non-original prompt variant is needed "
                    "for sensitivity analysis."
                )
        else:
            st.info(
                "Original prompt results are needed for sensitivity analysis."
            )


        # -------------------------------------------------
        # 5.5 Response similarity distribution
        # -------------------------------------------------

        st.subheader("5.5 Response Similarity Distribution")

        similarity_rows = []

        for question, group in plot_df.groupby("Question"):
            similarity = calculate_response_similarity(
                group["Model_Response"].dropna().astype(str).tolist()
            )

            if similarity is not None:
                similarity_rows.append({
                    "Question": question,
                    "Mean Response Similarity": similarity
                })

        distribution_similarity_df = pd.DataFrame(similarity_rows)

        if not distribution_similarity_df.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))

            ax.hist(
                distribution_similarity_df["Mean Response Similarity"],
                bins=10,
                edgecolor="black"
            )

            ax.set_xlabel("Mean TF-IDF cosine similarity")
            ax.set_ylabel("Number of questions")
            ax.set_title("Distribution of Response Similarity")
            ax.grid(axis="y", alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Higher similarity indicates more similar response text; "
                "it does not necessarily mean answers are correct."
            )
        else:
            st.info(
                "Response similarity could not be calculated for this run."
            )


        # -------------------------------------------------
        # 5.6 Response-time distribution
        # FIX: Do not pass labels or tick_labels into boxplot.
        # Set tick positions and labels separately for compatibility.
        # -------------------------------------------------

        st.subheader("5.6 Response-Time Distribution")

        latency_df = plot_df.dropna(
            subset=["Response_Time_Seconds"]
        )

        if not latency_df.empty:
            latency_data = []
            latency_labels = []

            for variant in variant_order:
                values = latency_df.loc[
                    latency_df["Variant"] == variant,
                    "Response_Time_Seconds"
                ].dropna().to_numpy()

                if len(values) > 0:
                    latency_data.append(values)
                    latency_labels.append(variant)

            if latency_data:
                fig, ax = plt.subplots(figsize=(11, 5))

                # Version-compatible Matplotlib call.
                ax.boxplot(
                    latency_data,
                    patch_artist=True
                )

                # Set labels independently to avoid the tick_labels error.
                ax.set_xticks(
                    list(range(1, len(latency_labels) + 1))
                )
                ax.set_xticklabels(
                    latency_labels,
                    rotation=30,
                    ha="right"
                )

                ax.set_ylabel("Response time (seconds)")
                ax.set_xlabel("Prompt variation")
                ax.set_title("Response-Time Variability by Prompt")
                ax.grid(axis="y", alpha=0.25)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No valid response-time measurements are available.")
        else:
            st.info("No valid response-time measurements are available.")


    # =====================================================
    # 13. MOST PROMPT-SENSITIVE QUESTIONS
    # =====================================================

    st.header("7. Most Prompt-Sensitive Questions")

    if differences is not None and not differences.empty:
        per_question_sensitivity = (
            differences.mean(axis=1)
            .sort_values(ascending=False)
        )

        sensitivity_df = (
            per_question_sensitivity
            .rename("Sensitivity Score")
            .reset_index()
        )

        sensitivity_df["Sensitivity Score"] = (
            sensitivity_df["Sensitivity Score"].round(3)
        )

        st.dataframe(
            sensitivity_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(per_question_sensitivity.head(10))
    else:
        st.info(
            "Sensitivity scores need Original results and at least one "
            "other prompt variant."
        )


    # =====================================================
    # 14. RESPONSE CONSISTENCY
    # =====================================================

    st.header("8. Response Consistency")

    if response_similarity:
        consistency_df = pd.DataFrame(response_similarity)

        st.dataframe(
            consistency_df,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.bar(
            range(len(consistency_df)),
            consistency_df["Mean Response Similarity"]
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean TF-IDF cosine similarity")
        ax.set_xlabel("Benchmark question")
        ax.set_title("Response Consistency by Question")

        ax.set_xticks(range(len(consistency_df)))
        ax.set_xticklabels(
            [
                str(q)[:35]
                for q in consistency_df["Question"]
            ],
            rotation=60,
            ha="right",
            fontsize=8
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info(
            "Not enough response text is available to calculate "
            "response similarity."
        )


    # =====================================================
    # 15. RESPONSE TIME ANALYSIS
    # =====================================================

    st.header("9. Response Time Analysis")

    response_time_by_variant = (
        df.groupby("Variant", sort=False)["Response_Time_Seconds"]
        .mean()
        .reindex(variant_order)
        .dropna()
    )

    if not response_time_by_variant.empty:
        st.bar_chart(response_time_by_variant)
    else:
        st.info("No response-time data is available.")

    st.caption(
        "Response time includes the local model request and generation time."
    )


    # =====================================================
    # 16. DOWNLOAD CURRENT RESULTS
    # =====================================================

    st.header("10. Download Current Results")

    st.download_button(
        "Download Benchmark CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prompt_sensitivity_benchmark.csv",
        mime="text/csv",
        key="download_benchmark"
    )


# =========================================================
# 17. PERSISTENT EXPERIMENT HISTORY
# =========================================================

st.markdown("---")
st.header("11. Experiment History")

history = load_history()

if history.empty:
    st.info(
        "No saved experiments yet. Run a custom experiment or benchmark "
        "to create history."
    )

else:
    st.write(f"Saved experiment records: {len(history)}")

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
            ["All Experiments"] + experiment_ids,
            key="history_experiment_filter"
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


# =========================================================
# 18. FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "PromptForge AI | Powered by Ollama | "
    "Intelligent LLM Evaluation Platform"
)