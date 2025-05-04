# BookBuddy: AI Book Recommendation Engine

An intelligent book recommendation system powered by LLaMA 3, LangChain, Gradio, and the Goodreads API.

## Features

- **Personalized Recommendations**: Get book suggestions based on your favorite books, genres, and reading preferences
- **Similar Book Discovery**: Find books similar to ones you already love
- **AI-Powered Analysis**: Uses LLaMA 3 to understand nuanced reading preferences and provide thoughtful recommendations
- **User-Friendly Interface**: Simple Gradio web interface for easy interaction

## How It Works

BookBuddy combines the power of large language models with literary data to understand your reading preferences and suggest books you'll enjoy. It does this by:

1. Analyzing your favorite books and genres to understand your literary taste
2. Using LLaMA 3 to process your preferences and match them with potential recommendations
3. Drawing from a database of books (with Goodreads API integration) to find the perfect matches
4. Providing personalized explanations for why each recommendation might appeal to you

## Technical Architecture

The system consists of several key components:

- **LLaMA 3**: State-of-the-art large language model for natural language understanding and generation
- **LangChain**: Framework for developing applications powered by language models
- **Gradio**: Web interface for user interaction
- **Book Database**: Local dataset with book information (with optional Goodreads API integration)

## Getting Started

### Prerequisites

- Python 3.8+
- LLaMA 3 model in GGUF format
- (Optional) Goodreads API key

### Installation

1. Clone this repository
   ```
   git clone https://github.com/yourusername/bookbuddy.git
   cd bookbuddy
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Download the LLaMA 3 model and place it in your project directory or specify its path in the environment variable
   ```
   export LLAMA_MODEL_PATH="/path/to/llama-3-8b-instruct.Q4_K_M.gguf"
   ```

4. (Optional) Set your Goodreads API key
   ```
   export GOODREADS_API_KEY="your_goodreads_api_key"
   ```

### Running the Application

1. Start the application
   ```
   python book_recommendation_engine.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

## Usage

### Get Personalized Recommendations

1. Enter your favorite books (comma-separated)
2. List your favorite genres
3. Describe your reading preferences (e.g., "I prefer shorter books with complex characters")
4. Click "Get Recommendations" to receive personalized suggestions

### Find Similar Books

1. Enter the title of a book you enjoy
2. Click "Find Similar Books" to discover related titles with explanations of their similarities

## Note on Goodreads API

The Goodreads API has been deprecated. The application includes a fallback mechanism using a local book database. For a production system, you might consider:

- Using web scraping with appropriate permissions
- Integrating with alternative book APIs (e.g., Google Books, OpenLibrary)
- Building your own comprehensive book database

## Future Enhancements

- Book cover image display
- User accounts and recommendation history
- Integration with e-reader platforms
- Collaborative filtering based on similar users
- Book availability at local libraries or bookstores

## License

This project is licensed under the MIT License - see the LICENSE file for details.
