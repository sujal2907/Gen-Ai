import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def getLLamaresponse(input_text, no_words, blog_style):
    model_path = 'D:/Article_generation/models/llama-2-7b-chat.ggmlv3.q8_0.bin'  # Use forward slashes for paths
    
    st.write(f"Model path: {model_path}")
    
    try:
        llm = CTransformers(model=model_path,
                            model_type='llama',
                            config={'max_new_tokens': 256,
                                    'temperature': 0.01})
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    st.write(f"Generated prompt: {formatted_prompt}")

    try:
        response = llm.invoke(formatted_prompt)
        st.write(f"Model response: {response}")
        return response
    except Exception as e:
        st.error(f"An error occurred during model invocation: {e}")
        return None

st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

if submit:
    if input_text and no_words:
        try:
            no_words = int(no_words)  # Convert to integer
            response = getLLamaresponse(input_text, no_words, blog_style)
            if response:
                st.write(response)
            else:
                st.warning("No response generated from the model.")
        except ValueError:
            st.error("Number of Words must be an integer.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both the topic and the number of words.")
