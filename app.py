import streamlit as st
import pandas as pd
import plotly.express as px

from src.helpers import analyze_csv, load_model, analyze_review

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Amazon Sentiment Insight",
    page_icon="🛒",
    layout="centered"
)

st.title("📦 Amazon Product Insight Engine")

tokenizer, model = load_model()

# Create Tabs
tab1, tab2 = st.tabs(["🔍 Single Review", "📂 Bulk CSV Analysis"])

# --- TAB 1: SINGLE REVIEW ---
with tab1:
    # We moved the logic that was at the bottom of your file INTO here
    if model is None:
        st.error(
            "❌ Model not found! Please ensure the path './roberta-base-amazon-finetuned/checkpoint-1875' is correct.")
    else:
        user_input = st.text_area("Paste a review here:", height=150,
                                  placeholder="Example: The battery life is amazing, but the screen scratches too easily.")

        if st.button("Analyze Sentiment"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    rating, conf, all_probs = analyze_review(user_input, tokenizer, model)

                st.divider()
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(label="Predicted Rating", value=f"{rating} / 5 ⭐")

                    # Color code based on rating
                    if rating >= 4:
                        st.success(f"Confidence: {conf:.1%}")
                    elif rating == 3:
                        st.warning(f"Confidence: {conf:.1%}")
                    else:
                        st.error(f"Confidence: {conf:.1%}")

                with col2:
                    # Create DataFrame for Plotly
                    chart_data = pd.DataFrame({
                        "Stars": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
                        "Probability": all_probs
                    })

                    fig = px.bar(
                        chart_data,
                        x="Stars",
                        y="Probability",
                        color="Probability",
                        color_continuous_scale="Blues",
                        range_y=[0, 1]
                    )
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please enter some text first.")

# --- TAB 2: BULK CSV ANALYSIS ---
with tab2:
    st.header("Upload Customer Data")
    st.write("Upload a CSV file containing a column named 'text' or 'review'.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # 1. GENERATE BUTTON LOGIC
    if uploaded_file is not None:
        if st.button("Generate Insight Report"):
            with st.spinner("Processing..."):
                # Run the heavy analysis
                df = analyze_csv(uploaded_file, model, tokenizer)

                # Save to Session State (The "Memory")
                st.session_state['results_df'] = df
                st.success("Analysis Complete!")

    # 2. DASHBOARD DISPLAY LOGIC
    # Check if data exists in memory (even if button wasn't just clicked)
    if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
        results_df = st.session_state['results_df']

        # --- DASHBOARD SECTION ---

        # 1. High-Level Metrics
        avg_rating = results_df["Predicted_Sentiment"].mean()
        total_reviews = len(results_df)
        pct_negative = (len(results_df[results_df["Predicted_Sentiment"] <= 2]) / total_reviews) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Rating", f"{avg_rating:.1f} / 5.0")
        col2.metric("Total Reviews", total_reviews)
        col3.metric("Negative Sentiment", f"{pct_negative:.1f}%", delta_color="inverse")

        st.divider()

        # 2. Strategic Advice Logic
        st.subheader("💡 Brand Strategy Recommendation")
        if avg_rating >= 4.5:
            st.success("Status: Market Leader. Customers are delighted.")
        elif avg_rating < 3.0:
            st.error("Status: Critical. High volume of defects detected.")
        elif pct_negative > 20:
            st.warning("Status: Polarizing. Significant defects driving customers away.")
        else:
            st.info("Status: Average. Look at 3-star reviews for improvements.")

        st.divider()

        # 3. Visuals
        st.subheader("Sentiment Distribution")
        sentiment_counts = results_df["Predicted_Sentiment"].value_counts().sort_index()

        bar_chart_data = pd.DataFrame({
            "Stars": ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
            "Count": [sentiment_counts.get(i, 0) for i in range(1, 6)]
        })
        st.bar_chart(bar_chart_data.set_index("Stars"))

        # 4. Deep Dive (THIS WILL NOW WORK)
        st.subheader("🔍 Voice of the Customer")

        # Changing this box triggers a rerun, but 'results_df' is safe in session_state!
        filter_rating = st.selectbox("Filter by Rating:", [1, 2, 3, 4, 5])

        filtered_reviews = results_df[results_df["Predicted_Sentiment"] == filter_rating]

        if not filtered_reviews.empty:
            for i, row in filtered_reviews.head(3).iterrows():
                # Find the text column again for display
                possible_cols = ["text", "review", "review_body", "content", "comment"]
                text_col = next((c for c in results_df.columns if c.lower() in possible_cols), results_df.columns[0])
                st.text_area(f"Sample Review #{i}", row[text_col], height=100, disabled=True)
        else:
            st.write("No reviews found for this rating.")