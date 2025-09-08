import streamlit as st
import pandas as pd
from openai import AzureOpenAI
import openai
from pydantic import BaseModel, confloat
import time
from tqdm import tqdm
import json
import requests
import mysql.connector
from mysql.connector import Error

# Set page configuration
st.set_page_config(
    page_title="Multi-Language Translation Evaluation Tool",
    page_icon="🌐",
    layout="wide"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'translated' not in st.session_state:
    st.session_state.translated = False
if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'translations' not in st.session_state:
    st.session_state.translations = {}
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = {}

# App title and description
st.title("🌐 Translation Evaluation Tool")
st.markdown("""
This app helps you evaluate translation quality by:
1. Connecting to a MySQL database with job descriptions from different countries
2. Fetching data from language-specific tables
3. Translating the descriptions using Azure OpenAI and/or Ollama models
4. Evaluating the translations against reference columns
""")

# Instructions
with st.expander("Instructions"):
    st.markdown("""
    ### How to Use This App
    
    1. **Database Connection**: 
       - Configure your MySQL database connection in the sidebar
       - Select the language/country for the data you want to analyze
       - Click "Connect to Database" to establish connection
    2. **Translation**: 
       - Configure Azure OpenAI settings
       - Optionally enable and configure Ollama for additional translations
       - Click the translation buttons to generate translations
    3. **Evaluation**: 
       - Select a reference column (correct translation)
       - Select a candidate column (to evaluate)
       - Click "Run Evaluation" to compare the translations
    4. **Download Results**: Download the evaluation results as CSV files
    
    ### Ollama Setup Instructions
    
    If you want to use Ollama models:
    
    1. Install Ollama from https://ollama.ai/
    2. Download the models you want to use:
       ```
       ollama pull phi-4
       ollama pull gpt_oss
       ollama pull eurollm-9b
       ```
    3. Start the Ollama server (usually runs automatically after installation)
    4. Test the connection using the button in the sidebar
    """)

# Notes
st.info("""
**Note**: 
- For evaluation, we use Azure OpenAI's GPT-4o model to ensure consistent and high-quality assessment
- Make sure you have sufficient API quota for the models you're using
- Ollama models run locally on your machine, so no internet connection is needed for translation
- Empty descriptions will be skipped during translation and evaluation
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Database Configuration")
    
    # Database connection settings
    db_host = st.text_input("Database Host", value="195.251.12.136")
    db_user = st.text_input("Database User", value="root")
    db_password = st.text_input("Database Password", type="password", value="dm@lab172")
    db_name = st.text_input("Database Name", value="jobsposting_sample")
    db_table = st.text_input("Table Name", value="Greek_Portals_RT")
    
    
    # Connect to database button
    if st.button("Connect to Database"):
        try:
            # Establish database connection
            connection = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name
            )
            
            if connection.is_connected():
                st.session_state.db_connection = connection
                st.session_state.db_connected = True
                st.success(f"✅ Successfully connected to the database!")
                
                # Fetch data from the database
                cursor = connection.cursor(dictionary=True)
                query = f"SELECT * FROM {db_table} limit 10"
                cursor.execute(query)
                results = cursor.fetchall()
                
                if results:
                    df = pd.DataFrame(results)
                    st.session_state.df = df
                    st.success(f"Fetched {len(df)} records from the table!")
                    
                    # Show data preview
                    with st.expander("Data Preview"):
                        st.dataframe(df.head())
                else:
                    st.warning("No data found in the specified table.")
                    
        except Error as e:
            st.error(f"❌ Database connection error: {e}")
            st.session_state.db_connected = False
    
    # Translation configuration
    st.header("Translation Configuration")
    
    # Azure OpenAI configuration
    st.subheader("Azure OpenAI Settings")
    azure_api_key = st.text_input("Azure API Key", type="password", 
                                 value="45KAZBei7wyoEoM0Ac3YGpKlDRvEEtLXrH9kqBkoHMOYmwCqA7YIJQQJ99ALACfhMk5XJ3w3AAAAACOGidRc")
    azure_endpoint = st.text_input("Azure Endpoint", 
                                  value="https://ai-viennas5522ai344254808993.openai.azure.com/")
    azure_api_version = st.text_input("API Version", value="2025-01-01-preview")
    
    # Azure model settings
    azure_model_name = st.selectbox("Azure Model", ["gpt-4o", "gpt-4-turbo", "gpt-35-turbo"])
    azure_temperature = st.slider("Azure Temperature", 0.0, 1.0, 0.0, 0.1)
    
    # Ollama configuration
    st.subheader("Ollama Settings")
    use_ollama = st.checkbox("Use Ollama for additional translation", value=False)
    
    if use_ollama:
        ollama_base_url = st.text_input("Ollama Base URL", value="http://195.251.12.136:11434/v1")
        ollama_api_key = st.text_input("Ollama API Key", value="ollama", type="password")
        
        # Ollama model selection
        ollama_model = st.selectbox("Ollama Model", ["phi4:14b", "gpt-oss", "eurollm-9b"])
        ollama_temperature = st.slider("Ollama Temperature", 0.0, 1.0, 0.0, 0.1)
        
        # Test Ollama connection
        if st.button("Test Ollama Connection"):
            try:
                response = requests.get(ollama_base_url.replace('/v1', '/api/tags'))
                if response.status_code == 200:
                    st.success("✅ Ollama connection successful!")
                    models = [model['name'] for model in response.json().get('models', [])]
                    st.write("Available models:", models)
                else:
                    st.error(f"❌ Ollama connection failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Ollama connection error: {e}")
    
    # Column names
    st.subheader("Column Names")
    text_column = st.text_input("Text Column", value="description")
    candidate_column = st.text_input("Candidate Column", value="en_description")
    id_column = st.text_input("ID Column", value="id")

# Data section
st.header("1. Database Data")
if st.session_state.db_connected and st.session_state.df is not None:
    st.success(f"Database connection active. Loaded {len(st.session_state.df)} records from table.")
    
    # Show data statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(st.session_state.df))
    with col2:
        st.metric("Columns", len(st.session_state.df.columns))
    with col3:
        # Count non-empty descriptions
        non_empty = st.session_state.df[text_column].notna().sum()
        st.metric("Non-empty Descriptions", non_empty)
    
    # Show data preview
    with st.expander("View Data Preview"):
        st.dataframe(st.session_state.df.head(10))
        
    # Show column information
    with st.expander("View Column Information"):
        st.write("Columns in the dataset:")
        for col in st.session_state.df.columns:
            st.write(f"- {col} (dtype: {st.session_state.df[col].dtype})")
            
    # Show sample descriptions
    with st.expander("View Sample Descriptions"):
        sample_descriptions = st.session_state.df[text_column].dropna().head(3).tolist()
        for i, desc in enumerate(sample_descriptions):
            st.write(f"**Sample {i+1}:**")
            st.text(desc[:200] + "..." if len(desc) > 200 else desc)
            st.write("---")
elif not st.session_state.db_connected:
    st.info("Please configure and connect to the database using the sidebar.")
else:
    st.info("No data loaded. Please connect to the database using the sidebar.")

# Translation section
st.header("2. Translation")
if st.session_state.df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Translate with Azure OpenAI", type="primary"):
            try:
                client = AzureOpenAI(
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint
                )
                
                
                # Translation function for Azure OpenAI
                def translate_text_azure(text, model=azure_model_name, temperature=azure_temperature):
                    if pd.isna(text) or text.strip() == "":
                        return ""
                    
                    prompt = f"""
                    Translate the following Greek text to English **directly and accurately**.
                    Greek Text:
                    {text}

                    English Translation:
                    """
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"Translation error: {e}")
                        return ""
                
                # Apply translation
                progress_bar = st.progress(0)
                status_text = st.empty()
                translated_texts = []
                
                for i, row in st.session_state.df.iterrows():
                    status_text.text(f"Translating row {i + 1}/{len(st.session_state.df)} with Azure")
                    translated = translate_text_azure(row[text_column])
                    translated_texts.append(translated)
                    progress_bar.progress((i + 1) / len(st.session_state.df))
                
                # Add translations to dataframe
                output_column = f"en_description_azure_{azure_model_name.replace('-', '_')}"
                st.session_state.df[output_column] = translated_texts
                
                # Store translation info
                if 'azure' not in st.session_state.translations:
                    st.session_state.translations['azure'] = []
                st.session_state.translations['azure'].append(output_column)
                
                status_text.text("Azure translation complete!")
                st.success(f"All descriptions have been translated successfully using Azure OpenAI {azure_model_name}.")
                
            except Exception as e:
                st.error(f"Error during Azure translation: {e}")
    
    with col2:
        if use_ollama and st.button("Translate with Ollama", type="primary"):
            try:
                # Initialize Ollama client
                open_client = openai.OpenAI(
                    base_url=ollama_base_url,
                    api_key=ollama_api_key,
                    timeout=520,
                )
                
                
                # Translation function for Ollama
                def translate_text_ollama(text, model=ollama_model, temperature=ollama_temperature):
                    if pd.isna(text) or text.strip() == "":
                        return ""
                    
                    prompt = f"""
                    Translate the following Greek text to English **directly and accurately**.
                    Greek Text:
                    {text}

                    English Translation:
                    """
                    try:
                        response = open_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"Translation error: {e}")
                        return ""
                
                # Apply translation
                progress_bar = st.progress(0)
                status_text = st.empty()
                translated_texts = []
                
                for i, row in st.session_state.df.iterrows():
                    status_text.text(f"Translating row {i + 1}/{len(st.session_state.df)} with Ollama")
                    translated = translate_text_ollama(row[text_column])
                    translated_texts.append(translated)
                    progress_bar.progress((i + 1) / len(st.session_state.df))
                
                # Add translations to dataframe
                output_column = f"en_description_ollama_{ollama_model}"
                st.session_state.df[output_column] = translated_texts
                
                # Store translation info
                if 'ollama' not in st.session_state.translations:
                    st.session_state.translations['ollama'] = []
                st.session_state.translations['ollama'].append(output_column)
                
                status_text.text("Ollama translation complete!")
                st.success(f"All descriptions have been translated successfully using Ollama {ollama_model}.")
                
            except Exception as e:
                st.error(f"Error during Ollama translation: {e}")

# Show translation results
if st.session_state.translations:
    st.subheader("Translation Results")
    
    # Show all translation columns
    with st.expander("View All Translation Columns"):
        translation_cols = [id_column, text_column, candidate_column]
        for method in st.session_state.translations:
            for col in st.session_state.translations[method]:
                translation_cols.append(col)
        
        st.dataframe(st.session_state.df[translation_cols].head())

# Evaluation section
st.header("3. Evaluation")
if st.session_state.translations:
    # Define Metrics class
    class Metrics(BaseModel):
        semantic_accuracy: confloat(ge=1.0, le=5.0)
        fluency: confloat(ge=1.0, le=5.0)
        description: str
    
    # Create evaluation form
    with st.form("evaluation_form"):
        st.subheader("Evaluation Configuration")
        
        # Select reference column
        reference_options = []
        for method in st.session_state.translations:
            for col in st.session_state.translations[method]:
                reference_options.append((f"{method}_{col}", col))
        
        reference_col = st.selectbox(
            "Reference Column (Correct Translation)",
            options=[col for _, col in reference_options],
            help="Select the column that contains the correct translation to use as reference"
        )
        
        # Select candidate column
        candidate_options = [candidate_column]  # Original candidate column
        for method in st.session_state.translations:
            for col in st.session_state.translations[method]:
                candidate_options.append(col)
        
        candidate_col = st.selectbox(
            "Candidate Column (To Evaluate)",
            options=candidate_options,
            help="Select the column that contains the translation to evaluate"
        )
        
        # Submit button
        submitted = st.form_submit_button("Run Evaluation", type="primary")
    
    if submitted:
        # Initialize Azure client for evaluation
        try:
            client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
            
            # Evaluation function
            def evaluate_translation(reference: str, candidate: str) -> Metrics:
                # Skip evaluation if either text is empty
                if pd.isna(reference) or pd.isna(candidate) or reference.strip() == "" or candidate.strip() == "":
                    return Metrics(semantic_accuracy=0.0, fluency=0.0, description="Skipped: Empty text")
                
                prompt = f"""
                You are a professional translator evaluating two English translations of a job posting.
                The first translation (Reference) is correct. The second (Candidate) is from an automated tool.

                Evaluate ONLY the two dimensions below and follow the rubric.

                Scoring rules:
                - Scores are real numbers in [1.0, 5.0].
                - Prefer half-point granularity (e.g., 4.5) when helpful.
                - Ignore punctuation, casing, bullet numbering, and harmless synonyms.
                - Penalize only meaning-changing issues for semantic accuracy; grammar/style only for fluency.

                Rubric (Semantic Accuracy):
                5.0 fully equivalent; 4.5 near-exact; 4.0 small drift; 3.0 clear divergence; 2.0 major errors; 1.0 wrong.

                Rubric (Fluency):
                5.0 native-like; 4.5 tiny nits; 4.0 readable but a bit awkward; 3.0 several issues; 2.0 many issues; 1.0 broken.

                Return JSON only, matching this schema exactly:
                {{
                  "semantic_accuracy": <number>,
                  "fluency": <number>,
                  "description": "<1-3 sentence summary of key semantic differences>"
                }}

                ### Reference:
                {reference}

                ### Candidate:
                {candidate}
                """

                try:
                    # Use standard completion if parse is not available
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    
                    content = response.choices[0].message.content.strip()
                    # Extract JSON from response
                    if content.startswith("```json"):
                        content = content[7:-3]  # Remove ```json and ``` markers
                    
                    data = json.loads(content)
                    
                    # Create Metrics object
                    metrics = Metrics(
                        semantic_accuracy=round(float(data["semantic_accuracy"]) * 2) / 2,
                        fluency=round(float(data["fluency"]) * 2) / 2,
                        description=data["description"]
                    )
                    
                    return metrics
                    
                except Exception as e:
                    st.error(f"Error parsing evaluation response: {e}")
                    return Metrics(semantic_accuracy=0.0, fluency=0.0, description=f"Error: {str(e)}")
            
            # Evaluate all rows
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in st.session_state.df.iterrows():
                status_text.text(f"Evaluating row {i + 1}/{len(st.session_state.df)}")
                
                try:
                    metrics = evaluate_translation(
                        reference=row[reference_col],
                        candidate=row[candidate_col]
                    )
                    results.append({
                        id_column: row[id_column],
                        "semantic_accuracy": metrics.semantic_accuracy,
                        "fluency": metrics.fluency,
                        "description": metrics.description,
                        "reference_translation": row[reference_col],
                        "candidate_translation": row[candidate_col],
                        "reference_column": reference_col,
                        "candidate_column": candidate_col
                    })
                except Exception as e:
                    st.error(f"Error evaluating ID {row[id_column]}: {e}")
                    results.append({
                        id_column: row[id_column],
                        "semantic_accuracy": None,
                        "fluency": None,
                        "description": None,
                        "reference_translation": row[reference_col],
                        "candidate_translation": row[candidate_col],
                        "reference_column": reference_col,
                        "candidate_column": candidate_col
                    })
                
                progress_bar.progress((i + 1) / len(st.session_state.df))
                time.sleep(0.1)  # Small delay to avoid rate limiting
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Store evaluation results
            eval_key = f"{reference_col}_vs_{candidate_col}"
            st.session_state.evaluations[eval_key] = results_df
            
            status_text.text("Evaluation complete!")
            st.success(f"Evaluation of {candidate_col} against {reference_col} completed successfully.")
            
            # Show evaluation results
            with st.expander("View Evaluation Results"):
                st.dataframe(results_df)
                
            # Show summary statistics
            with st.expander("View Summary Statistics"):
                if not results_df.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Filter out skipped evaluations
                        valid_sa = results_df[results_df['semantic_accuracy'] > 0]['semantic_accuracy']
                        valid_fluency = results_df[results_df['fluency'] > 0]['fluency']
                        
                        if len(valid_sa) > 0:
                            st.metric("Average Semantic Accuracy", f"{valid_sa.mean():.2f}")
                            st.metric("Minimum Semantic Accuracy", f"{valid_sa.min():.2f}")
                        else:
                            st.write("No valid semantic accuracy scores")
                    
                    with col2:
                        if len(valid_fluency) > 0:
                            st.metric("Average Fluency", f"{valid_fluency.mean():.2f}")
                            st.metric("Minimum Fluency", f"{valid_fluency.min():.2f}")
                        else:
                            st.write("No valid fluency scores")
                    
                    with col3:
                        if len(valid_sa) > 0 and len(valid_fluency) > 0:
                            # Count of translations with high quality (score >= 4.0)
                            high_quality_sa = len(valid_sa[valid_sa >= 4.0])
                            high_quality_fluency = len(valid_fluency[valid_fluency >= 4.0])
                            
                            st.metric("High Quality Semantic Accuracy (≥4.0)", 
                                     f"{high_quality_sa} ({high_quality_sa/len(valid_sa)*100:.1f}%)")
                            st.metric("High Quality Fluency (≥4.0)", 
                                     f"{high_quality_fluency} ({high_quality_fluency/len(valid_fluency)*100:.1f}%)")
                
        except Exception as e:
            st.error(f"Error during evaluation: {e}")

# Show all evaluation results
if st.session_state.evaluations:
    st.header("4. All Evaluation Results")
    
    for eval_key in st.session_state.evaluations:
        with st.expander(f"Evaluation: {eval_key}"):
            st.dataframe(st.session_state.evaluations[eval_key])
            
            # Summary for this evaluation
            results_df = st.session_state.evaluations[eval_key]
            valid_sa = results_df[results_df['semantic_accuracy'] > 0]['semantic_accuracy']
            valid_fluency = results_df[results_df['fluency'] > 0]['fluency']
            
            if len(valid_sa) > 0 and len(valid_fluency) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Semantic Accuracy", f"{valid_sa.mean():.2f}")
                with col2:
                    st.metric("Avg Fluency", f"{valid_fluency.mean():.2f}")



# Close database connection when done
if st.session_state.db_connected:
    if st.button("Disconnect from Database"):
        try:
            st.session_state.db_connection.close()
            st.session_state.db_connected = False
            st.success("Disconnected from database.")
        except:
            st.error("Error disconnecting from database.")
