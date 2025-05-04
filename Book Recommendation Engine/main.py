import os
import re
import requests
import gradio as gr
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import xml.etree.ElementTree as ET

# Configuration
GOODREADS_API_KEY = os.environ.get("GOODREADS_API_KEY", "your_goodreads_api_key")
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "./llama-3-8b-instruct.Q4_K_M.gguf")

# Initialize the LLaMA model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=LLAMA_MODEL_PATH,
    temperature=0.7,
    max_tokens=2000,
    top_p=0.95,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=4096  # Context window size
)

class BookRecommender:
    def __init__(self, llm, goodreads_api_key):
        self.llm = llm
        self.goodreads_api_key = goodreads_api_key
        self.prompt_template = self._create_prompt_template()
        self.recommendation_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        # Sample book database (in a real app, you might use a larger dataset)
        self.book_database = self._load_book_database()
    
    def _load_book_database(self) -> pd.DataFrame:
        """
        Load a sample book database or create one.
        In a production system, this would be a larger dataset.
        """
        # This is a small sample database; in production you'd use more data
        data = {
            'title': [
                'The Hobbit', '1984', 'To Kill a Mockingbird', 'Pride and Prejudice',
                'The Great Gatsby', 'The Catcher in the Rye', 'Brave New World',
                'The Lord of the Rings', 'Crime and Punishment', 'The Odyssey',
                'Fahrenheit 451', 'Jane Eyre', 'Moby Dick', 'Wuthering Heights',
                'The Alchemist', 'War and Peace', 'The Divine Comedy', 'Hamlet',
                'Don Quixote', 'The Brothers Karamazov', 'Sapiens', 'Thinking, Fast and Slow',
                'Educated', 'The Power of Habit', 'Atomic Habits', 'The Subtle Art of Not Giving a F*ck',
                'Project Hail Mary', 'The Midnight Library', 'Where the Crawdads Sing', 'Klara and the Sun'
            ],
            'author': [
                'J.R.R. Tolkien', 'George Orwell', 'Harper Lee', 'Jane Austen',
                'F. Scott Fitzgerald', 'J.D. Salinger', 'Aldous Huxley',
                'J.R.R. Tolkien', 'Fyodor Dostoevsky', 'Homer',
                'Ray Bradbury', 'Charlotte Brontë', 'Herman Melville', 'Emily Brontë',
                'Paulo Coelho', 'Leo Tolstoy', 'Dante Alighieri', 'William Shakespeare',
                'Miguel de Cervantes', 'Fyodor Dostoevsky', 'Yuval Noah Harari', 'Daniel Kahneman',
                'Tara Westover', 'Charles Duhigg', 'James Clear', 'Mark Manson',
                'Andy Weir', 'Matt Haig', 'Delia Owens', 'Kazuo Ishiguro'
            ],
            'genre': [
                'Fantasy', 'Dystopian', 'Literary Fiction', 'Classic',
                'Classic', 'Coming-of-age', 'Dystopian',
                'Fantasy', 'Philosophical Fiction', 'Epic',
                'Dystopian', 'Gothic Fiction', 'Adventure', 'Gothic Fiction',
                'Philosophical Fiction', 'Historical Fiction', 'Epic Poetry', 'Tragedy',
                'Satire', 'Philosophical Fiction', 'Non-fiction', 'Psychology',
                'Memoir', 'Psychology', 'Self-help', 'Self-help',
                'Science Fiction', 'Fantasy', 'Mystery', 'Science Fiction'
            ],
            'description': [
                'A hobbit sets out on an adventure with dwarves to reclaim their treasure from a dragon.',
                'A man rebels against a totalitarian government in a dystopian future.',
                'A young girl observes her father defend a black man accused of rape in the Depression-era South.',
                'A story of love and misunderstanding among the landed gentry in Georgian England.',
                'A mysterious millionaire's obsession leads to tragedy in the Jazz Age.',
                'A teenager's angst-filled wanderings in New York City after being expelled from prep school.',
                'A future society where people are engineered for specific roles and kept happy with drugs.',
                'An epic quest to destroy a ring of power and defeat the Dark Lord.',
                'A man's psychological turmoil after committing murder.',
                'The hero's journey home after the Trojan War.',
                'In a future society, books are banned and "firemen" burn them.',
                'An orphan girl's journey to adulthood and romance with the enigmatic Mr. Rochester.',
                'A sailor's obsessive hunt for the white whale that took his leg.',
                'A passionate and cruel story of love and revenge on the Yorkshire moors.',
                'A shepherd boy travels to Egypt after a recurring dream about finding treasure there.',
                'A saga of Russian society during the Napoleonic Wars.',
                'A journey through Hell, Purgatory, and Paradise.',
                'A prince seeks revenge for his father's murder.',
                'A man becomes so immersed in reading chivalric romances he loses his sanity.',
                'A philosophical novel that explores faith, doubt, and morality.',
                'A brief history of humankind from the Stone Age to the present.',
                'Explores the two systems of thinking that drive how we think and make choices.',
                'A memoir about growing up in a survivalist family and the pursuit of education.',
                'Examines how habits work and how they can be changed.',
                'Small habits can lead to remarkable results in self-improvement.',
                'A counterintuitive approach to living a good life by embracing life's struggles.',
                'An astronaut wakes up with amnesia and must save humanity from extinction.',
                'A library between life and death where each book represents a different life path.',
                'A girl who grew up isolated in the marshes becomes a murder suspect.',
                'An artificial "friend" observes and tries to understand the human condition.'
            ],
            'popularity': [
                85, 92, 88, 80, 86, 79, 78, 95, 82, 75, 
                83, 77, 72, 74, 89, 76, 71, 84, 81, 78, 
                90, 87, 84, 83, 92, 88, 91, 85, 89, 82
            ],
            'similar_books': [
                'The Lord of the Rings,The Chronicles of Narnia', '1984,Brave New World,Fahrenheit 451', 'To Kill a Mockingbird,The Color Purple',
                'Pride and Prejudice,Sense and Sensibility,Emma', 'The Great Gatsby,Tender Is the Night', 'The Catcher in the Rye,The Bell Jar',
                'Brave New World,1984,Fahrenheit 451', 'The Lord of the Rings,The Hobbit,A Game of Thrones', 'Crime and Punishment,The Idiot',
                'The Odyssey,The Iliad', 'Fahrenheit 451,1984,Brave New World', 'Jane Eyre,Wuthering Heights',
                'Moby Dick,The Old Man and the Sea', 'Wuthering Heights,Jane Eyre', 'The Alchemist,The Little Prince',
                'War and Peace,Anna Karenina', 'The Divine Comedy,Paradise Lost', 'Hamlet,Macbeth,King Lear',
                'Don Quixote,Gulliver\'s Travels', 'The Brothers Karamazov,Crime and Punishment', 'Sapiens,Guns, Germs, and Steel',
                'Thinking, Fast and Slow,Predictably Irrational', 'Educated,The Glass Castle', 'The Power of Habit,Atomic Habits',
                'Atomic Habits,The Power of Habit', 'The Subtle Art of Not Giving a F*ck,Everything Is F*cked',
                'Project Hail Mary,The Martian', 'The Midnight Library,The Invisible Life of Addie LaRue', 'Where the Crawdads Sing,The Great Alone',
                'Klara and the Sun,Never Let Me Go'
            ]
        }
        return pd.DataFrame(data)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for the LLM to generate book recommendations."""
        template = """
        You are an expert book recommendation system named BookBuddy. You make thoughtful book 
        recommendations based on users' reading history and preferences.
        
        USER'S FAVORITE BOOKS: {favorite_books}
        USER'S FAVORITE GENRES: {favorite_genres}
        USER'S READING PREFERENCES: {reading_preferences}
        BOOKS TO RECOMMEND FROM: {available_books}
        
        TASK:
        1. Analyze the user's favorite books, genres, and reading preferences
        2. Find 5 books from the AVAILABLE BOOKS that best match their interests
        3. For each recommendation, explain why it's a good match for this specific user
        4. Format your response in a friendly, conversational way
        
        Your recommendations should be specific to this user's tastes. Provide thoughtful, personalized explanations.
        """
        return PromptTemplate(
            input_variables=["favorite_books", "favorite_genres", "reading_preferences", "available_books"],
            template=template
        )
    
    def search_goodreads(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for books using the Goodreads API.
        
        Args:
            query: The search query
            
        Returns:
            A list of book data dictionaries
        """
        try:
            # Format the URL for the Goodreads API search
            url = f"https://www.goodreads.com/search/index.xml?key={self.goodreads_api_key}&q={query}"
            
            # Make the request
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error querying Goodreads API: {response.status_code}")
                return []
            
            # Parse the XML response
            root = ET.fromstring(response.content)
            
            # Extract book data
            books = []
            for work in root.findall('.//work'):
                book = {}
                book['id'] = work.find('id').text if work.find('id') is not None else 'Unknown'
                book['title'] = work.find('./best_book/title').text if work.find('./best_book/title') is not None else 'Unknown'
                book['author'] = work.find('./best_book/author/name').text if work.find('./best_book/author/name') is not None else 'Unknown'
                book['rating'] = work.find('average_rating').text if work.find('average_rating') is not None else 'Unknown'
                books.append(book)
            
            return books[:10]  # Return at most 10 books
        except Exception as e:
            print(f"Error searching Goodreads: {e}")
            return []
    
    def get_book_details(self, book_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a book using the Goodreads API.
        
        Args:
            book_id: The Goodreads book ID
            
        Returns:
            A dictionary containing book details
        """
        # In a real implementation, this would call the Goodreads API
        # Since the Show API has been discontinued, we're simulating data
        # You might want to use web scraping or another API as an alternative
        
        # Using our local database instead
        book = self.book_database[self.book_database['title'].str.contains(book_id, case=False)].iloc[0] \
            if len(self.book_database[self.book_database['title'].str.contains(book_id, case=False)]) > 0 else None
        
        if book is not None:
            return {
                'title': book['title'],
                'author': book['author'],
                'genre': book['genre'],
                'description': book['description'],
                'similar_books': book['similar_books'].split(',')
            }
        return {
            'title': book_id,
            'author': 'Unknown',
            'genre': 'Unknown',
            'description': 'No description available',
            'similar_books': []
        }
    
    def filter_books_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """
        Filter books by genre from our local database.
        
        Args:
            genre: The genre to filter by
            
        Returns:
            A list of books in that genre
        """
        filtered = self.book_database[self.book_database['genre'].str.contains(genre, case=False)]
        return filtered.to_dict('records')
    
    def generate_recommendations(self, favorite_books: str, favorite_genres: str, 
                                reading_preferences: str) -> str:
        """
        Generate book recommendations based on user preferences.
        
        Args:
            favorite_books: User's favorite books (comma-separated)
            favorite_genres: User's favorite genres (comma-separated)
            reading_preferences: User's preferences (e.g., length, complexity)
            
        Returns:
            Recommendations as formatted text
        """
        # Format available books for the prompt
        available_books_text = "\n".join([
            f"- '{book['title']}' by {book['author']} ({book['genre']}): {book['description'][:100]}..."
            for _, book in self.book_database.sample(min(15, len(self.book_database))).iterrows()
        ])
        
        # Get recommendations from LLM
        recommendations = self.recommendation_chain.run(
            favorite_books=favorite_books,
            favorite_genres=favorite_genres,
            reading_preferences=reading_preferences,
            available_books=available_books_text
        )
        
        return recommendations
    
    def find_similar_books(self, book_title: str) -> str:
        """
        Find books similar to the given title.
        
        Args:
            book_title: The title of the book to find similar books for
            
        Returns:
            Text containing similar book recommendations
        """
        # Try to find the book in our database
        matched_books = self.book_database[self.book_database['title'].str.contains(book_title, case=False)]
        
        if len(matched_books) > 0:
            book = matched_books.iloc[0]
            similar_books = book['similar_books'].split(',')
            
            # Create a prompt for the LLM to explain the similarities
            prompt = f"""
            I'm looking for books similar to '{book_title}' by {book['author']}.
            
            Here are some books that are considered similar:
            {', '.join(similar_books)}
            
            Please explain why these books are similar to '{book_title}' and which one I should read next
            based on thematic elements, writing style, and reader experience. Give your explanation in
            a friendly, conversational tone.
            """
            
            return llm.predict(prompt)
        else:
            # If not in our database, use a generic prompt
            prompt = f"""
            I'm looking for books similar to '{book_title}'.
            
            Based on your knowledge of literature, what books would you recommend that are similar to 
            '{book_title}' in terms of themes, writing style, or overall reading experience?
            
            Please recommend 3-5 books and explain why they're similar. Give your explanation in 
            a friendly, conversational tone.
            """
            
            return llm.predict(prompt)


class BookRecommendationApp:
    def __init__(self):
        """Initialize the Gradio app for book recommendations."""
        self.recommender = BookRecommender(llm, GOODREADS_API_KEY)
        self.app = self._build_interface()
    
    def _build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(title="BookBuddy - AI Book Recommendation System") as app:
            gr.Markdown("# BookBuddy: AI Book Recommendation System")
            gr.Markdown("Powered by LLaMA 3, LangChain, and Goodreads")
            
            with gr.Tab("Get Personalized Recommendations"):
                with gr.Row():
                    with gr.Column():
                        favorite_books = gr.Textbox(
                            label="Your Favorite Books (comma-separated)",
                            placeholder="e.g., The Great Gatsby, 1984, Pride and Prejudice"
                        )
                        favorite_genres = gr.Textbox(
                            label="Your Favorite Genres (comma-separated)",
                            placeholder="e.g., Science Fiction, Mystery, Fantasy"
                        )
                        reading_preferences = gr.Textbox(
                            label="Reading Preferences",
                            placeholder="e.g., I prefer shorter books with complex characters and unexpected endings"
                        )
                        submit_button = gr.Button("Get Recommendations")
                    
                    with gr.Column():
                        recommendations_output = gr.Markdown(label="Your Personalized Recommendations")
            
            with gr.Tab("Find Similar Books"):
                with gr.Row():
                    with gr.Column():
                        book_title_input = gr.Textbox(
                            label="Book Title",
                            placeholder="Enter a book title to find similar books"
                        )
                        similar_button = gr.Button("Find Similar Books")
                    
                    with gr.Column():
                        similar_books_output = gr.Markdown(label="Similar Books")
            
            # Connect the components
            submit_button.click(
                fn=self.recommender.generate_recommendations,
                inputs=[favorite_books, favorite_genres, reading_preferences],
                outputs=recommendations_output
            )
            
            similar_button.click(
                fn=self.recommender.find_similar_books,
                inputs=[book_title_input],
                outputs=similar_books_output
            )
            
            return app
    
    def launch(self):
        """Launch the Gradio app."""
        return self.app.launch()


if __name__ == "__main__":
    app = BookRecommendationApp()
    app.launch()
