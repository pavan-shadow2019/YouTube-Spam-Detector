import json
import requests
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Scrollbar
import os
import pickle
import re
import joblib
import datetime
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import io


# Set the console encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

# Define your YouTube API credentials and scope
YOUTUBE_API_KEY_FILE = "api_key.txt"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_API_PARAMS = {
    "part": "snippet",
    "maxResults": 100,
    "key": None,  # API key will be loaded from api_key.txt
    "textFormat": "plainText"
}

SPAM_MODEL_FILE = "stacked_model.pkl"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
TFIDF_VOCABULARY_FILE = "tfidf_vocabulary.pkl"

class YouTubeSpamDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Spam Detection Tool")
        self.root.geometry("800x600")
        self.current_frame = None
        self.selected_option = tk.StringVar()
        self.logo_image = tk.PhotoImage(file="logo_ytspam.png")
        self.create_welcome_frame()
        self.comments_text = None
        self.tfidf_vect = None
        
    def create_welcome_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Create a left frame for the logo
        left_frame = tk.Frame(self.current_frame)
        left_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        # Display the logo on the left side (take left half of the space)
        image_label = tk.Label(left_frame, image=self.logo_image)
        image_label.image = self.logo_image
        image_label.pack(fill=tk.BOTH, expand=True)

        # Create a right frame for options
        right_frame = tk.Frame(self.current_frame)
        right_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        label = tk.Label(right_frame, text="Welcome to YouTube Spam Detection Tool", font=("Arial", 16))
        label.pack(pady=21)

        intro_text = """This tool helps you analyze and filter spam comments on your YouTube channel.\n\nClick 'Start' to begin."""
        intro_label = tk.Label(right_frame, text=intro_text, justify=tk.LEFT)
        intro_label.pack(pady=2)

        start_button = tk.Button(right_frame, text="Start", command=self.create_option_selection_frame)
        start_button.pack()

    def create_option_selection_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Create a left frame for the logo
        left_frame = tk.Frame(self.current_frame)
        left_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        # Display the logo on the left side (take left half of the space)
        image_label = tk.Label(left_frame, image=self.logo_image)
        image_label.image = self.logo_image
        image_label.pack(fill=tk.BOTH, expand=True)

        # Create a right frame for options
        right_frame = tk.Frame(self.current_frame)
        right_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        option_description = tk.Label(right_frame, text="Please select an option from the dropdown below:")
        option_description.pack()

        option_menu = ttk.Combobox(right_frame, textvariable=self.selected_option, values=["Scan Your Channel Video", "Scan Others' Channel Video", "Help & Tips"], height=3)
        option_menu.pack()

        submit_button = tk.Button(right_frame, text="Submit", command=self.handle_option_selection)
        submit_button.pack()

    def handle_option_selection(self):
        selected_option = self.selected_option.get()
        if selected_option == "Scan Your Channel Video":
            self.create_video_input_frame()
        elif selected_option == "Scan Others' Channel Video":
            pass
        elif selected_option == "Help & Tips":
            pass
        else:
            messagebox.showerror("Invalid Option", "Please select a valid option.")

    def create_video_input_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Create a left frame for the logo
        left_frame = tk.Frame(self.current_frame)
        left_frame.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")

        # Display the logo on the left side (take left half of the space)
        image_label = tk.Label(left_frame, image=self.logo_image)
        image_label.image = self.logo_image
        image_label.pack(fill=tk.BOTH, expand=True)

        # Create a right frame for the input field
        right_frame = tk.Frame(self.current_frame)
        right_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        label = tk.Label(right_frame, text="Enter Video ID or Link")
        label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.video_id_entry = tk.Entry(right_frame, width=40)
        self.video_id_entry.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        submit_button = tk.Button(right_frame, text="Submit", command=self.fetch_and_display_comments)
        submit_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    def fetch_and_display_comments(self):
        video_id_or_link = self.video_id_entry.get()
        api_key = self.load_api_key()

        if not api_key:
            return

        try:
            video_title, comments_with_usernames = self.fetch_comments(api_key, video_id_or_link)

            # Check if there are comments before trying to process them
            if comments_with_usernames:
                spam_comments_with_usernames = self.detect_spam(comments_with_usernames)

                # Save comments and spam comments to CSV files
                self.save_comments_to_file(comments_with_usernames, "all_comments.csv")
                self.save_comments_to_file(spam_comments_with_usernames, "spam_comments.csv")

                # Prepare the live comments for classification
                live_comments_df = self.prepare_live_comments_dataframe(comments_with_usernames)
                print("Live comments DataFrame prepared.")

                live_comments_df = self.apply_imputation(live_comments_df)
                print("Imputation applied to live comments DataFrame.")

                # Classify live comments
                live_comments_spam = self.classify_live_comments(live_comments_df)
                print("Live comments classified.")

                # Display filtered spam comments in a new frame
                self.display_spam_comments_frame(
                    video_title,
                    live_comments_spam,
                    len(comments_with_usernames),
                    len(live_comments_spam)
                )
            else:
                print("No comments fetched from the video.")
        except Exception as e:
            print(f"Error details: {str(e)}")  # Log the details of the exception
            import traceback
            traceback.print_exc()  # Print the traceback
            messagebox.showerror("Error", f"An error occurred while fetching comments or performing spam detection: {str(e)}")
                     
    def get_video_title(self, video_id, api_key):
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet"
        response = requests.get(url)
        data = json.loads(response.text)
        title = data['items'][0]['snippet']['title']
        return title
                    
    def prepare_live_comments_dataframe(self, comments_with_usernames):
        # Create a DataFrame
        live_comments_df = pd.DataFrame(comments_with_usernames, columns=['Author', 'Comment'])
        return live_comments_df

    def apply_imputation(self, comments_df):
        if comments_df.empty:
            print("No comments found. Skipping imputation.")
            return comments_df

        imputer = SimpleImputer(strategy='most_frequent')
        comments_df['Comment'] = imputer.fit_transform(comments_df[['Comment']]).ravel()
        return comments_df


    def load_api_key(self):
        try:
            with open(YOUTUBE_API_KEY_FILE, "r") as api_key_file:
                api_key = api_key_file.read().strip()
            return api_key
        except FileNotFoundError:
            messagebox.showerror("API Key Not Found", "Please create an 'api_key.txt' file with your YouTube API key.")
            return None

    def fetch_comments(self, api_key, video_id_or_link):
        try:
            video_id = self.extract_video_id(video_id_or_link)
            if not video_id:
                messagebox.showerror("Invalid Video ID", "Please provide a valid Video ID or link.")
                return "", []

            api_params = dict(YOUTUBE_API_PARAMS)
            api_params["key"] = api_key
            api_params["videoId"] = video_id

            # Print the API parameters for debugging
            print(f"API Parameters: {api_params}")

            comments_with_usernames = []
            nextPageToken = None

            while True:
                if nextPageToken:
                    api_params["pageToken"] = nextPageToken

                response = requests.get(YOUTUBE_API_URL, params=api_params)

                # Print the full API response for debugging
                print(f"Full API Response: {response.json()}")

                response.raise_for_status()  # This will raise an HTTPError if the request failed
                data = response.json()

                # Check for errors in the API response
                if 'error' in data:
                    error_message = data['error']['message']
                    raise Exception(f"YouTube API error: {error_message}")

                for item in data.get("items", []):
                    print(f"item type: {type(item)}, item value: {item}")
                    author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
                    comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comment_with_username = {"Author": author, "Comment": comment_text}
                    comments_with_usernames.append(comment_with_username)

                nextPageToken = data.get("nextPageToken")

                if not nextPageToken:
                    break

            video_title = self.get_video_title(video_id, api_key)

            return video_title, comments_with_usernames

        except requests.exceptions.RequestException as e:
            print(f"Request Exception: {str(e)}")
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text}")
            messagebox.showerror("Network Error", "Unable to fetch comments. Please check your internet connection.")
            return "", []

        except Exception as e:
            print(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred while fetching comments: {str(e)}")
            return "", []




    def extract_video_id(self, video_id_or_link):
        # Check if the input is already a video ID (contains no slashes or query parameters)
        if re.fullmatch(r"[a-zA-Z0-9_-]+", video_id_or_link):
            return video_id_or_link

        # Extract video ID from YouTube URL
        video_id_match = re.search(r"(?<=v=)[^&#]+", video_id_or_link)
        video_id_match = video_id_match or re.search(r"(?<=be/)[^&#]+", video_id_or_link)
        video_id_match = video_id_match or re.search(r"(?<=embed/)[^&#]+", video_id_or_link)

        return video_id_match.group(0) if video_id_match else None


    def classify_live_comments(self, live_comments_df):
        if self.tfidf_vect is None:
            messagebox.showerror("TF-IDF Vectorizer Not Found", "Please load the TF-IDF vectorizer used during training.")
            return []

        live_comments_tfidf = self.tfidf_vect.transform(live_comments_df['Comment'])

        try:
            stacked_model = joblib.load(SPAM_MODEL_FILE)
        except FileNotFoundError:
            messagebox.showerror("Model Not Found", "Please make sure the model file is available.")
            return []

        live_comments_pred = stacked_model.predict(live_comments_tfidf)
        live_comments_spam = live_comments_df[live_comments_pred == 1]

        return live_comments_spam

    def detect_spam(self, comments_with_usernames):
        try:
            spam_detection_model = joblib.load(SPAM_MODEL_FILE)
        except FileNotFoundError:
            messagebox.showerror("File Not Found", "Please make sure the model file is available.")
            return []

        try:
            self.tfidf_vect = joblib.load(TFIDF_VECTORIZER_FILE)
            with open(TFIDF_VOCABULARY_FILE, 'rb') as vocab_file:
                self.tfidf_vect.vocabulary_ = pickle.load(vocab_file)
        except FileNotFoundError:
            messagebox.showerror("File Not Found", "Please make sure the TF-IDF vectorizer and vocabulary files are available.")
            return []

        spam_comments_with_usernames = []

        for comment_with_username in comments_with_usernames:
            author, comment_text = comment_with_username['Author'], comment_with_username['Comment']

            # Prepare the input data as a list of comment texts
            comment_texts = [comment_text]

            # Transform the comments into TF-IDF vectors
            comment_tfidf = self.tfidf_vect.transform(comment_texts)

            # Use the loaded model to predict if the comments are spam (1) or not (0)
            prediction = spam_detection_model.predict(comment_tfidf)

            if prediction[0] == 1:
                spam_comments_with_usernames.append({"Author": author, "Comment": comment_text})

        self.save_comments_to_file(comments_with_usernames, "all_comments.csv")

        return spam_comments_with_usernames
    
    def display_comments(self, comments):
        if self.comments_text:
            self.comments_text.destroy()

        comments_frame = tk.Frame(self.current_frame)
        comments_frame.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")

        self.comments_text = tk.Text(comments_frame, wrap=tk.WORD, height=20, width=60)
        self.comments_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = Scrollbar(comments_frame, command=self.comments_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comments_text.config(yscrollcommand=scrollbar.set)

        for comment in comments:
            self.comments_text.insert(tk.END, comment + "\n")
            
    def display_spam_comments_frame(self, video_title, spam_comments, total_comments, spam_detected):
        if self.current_frame:
            self.current_frame.destroy()

        self.current_frame = tk.Frame(self.root)
        self.current_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Create a Text widget to display spam comments
        spam_text = tk.Text(self.current_frame, wrap=tk.WORD, height=20, width=60)
        spam_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = Scrollbar(self.current_frame, command=spam_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        spam_text.config(yscrollcommand=scrollbar.set)

        for _, comment in spam_comments.iterrows():
            try:
                author = comment['Author']
                comment_text = comment['Comment']
                spam_text.insert(tk.END, f"Author: {author}\n")
                spam_text.insert(tk.END, f"Comment: {comment_text}\n\n")
            except Exception as e:
                print(f"Error while processing comment: {str(e)}")
                continue

        # Display video details, total comments, and spam detected
        video_details_label = tk.Label(self.current_frame, text=f"Video Title: {video_title}")
        video_details_label.pack()

        total_comments_label = tk.Label(self.current_frame, text=f"Total Comments Scanned: {total_comments}")
        total_comments_label.pack()

        spam_detected_label = tk.Label(self.current_frame, text=f"Spam Detected: {spam_detected}")
        spam_detected_label.pack()

    

    def save_comments_to_file(self, comments, filename):
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=["Author", "Comment"])
            csv_writer.writeheader()
            for comment in comments:
                csv_writer.writerow({"Author": comment['Author'], "Comment": comment['Comment']})

    def load_tfidf_vectorizer(self):
        try:
            self.tfidf_vect = joblib.load(TFIDF_VECTORIZER_FILE)
            print("TF-IDF vectorizer loaded successfully.")
        except FileNotFoundError:
            messagebox.showerror("File Not Found", "TF-IDF vectorizer file not found. Please create it first.")

    def load_tfidf_vocabulary(self):
        if self.tfidf_vect is not None:
            try:
                with open(TFIDF_VOCABULARY_FILE, 'rb') as vocab_file:
                    self.tfidf_vect.vocabulary_ = pickle.load(vocab_file)
                print("TF-IDF vocabulary loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("File Not Found", "TF-IDF vocabulary file not found. Please create it first.")
        else:
            messagebox.showerror("TF-IDF Vectorizer Not Loaded", "Please load the TF-IDF vectorizer first.")

    def run(self):
        self.load_tfidf_vectorizer()
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = YouTubeSpamDetectionApp(root)
    app.run()