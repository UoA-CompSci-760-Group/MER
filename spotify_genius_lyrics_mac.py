import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import logging
import random
import signal
import sys
import re
from datetime import datetime, timedelta
import json
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lyrics_fetcher.log"),
        logging.StreamHandler()
    ]
)

class SmartRateLimiter:    
    def __init__(self, default_pause=1.0, max_pause=5.0, 
                 hourly_limit=100, cooldown_factor=1.5):
        self.default_pause = default_pause
        self.max_pause = max_pause
        self.hourly_limit = hourly_limit
        self.cooldown_factor = cooldown_factor
        self.calls = []
        self.consecutive_errors = 0
        
    def sleep(self, error=False):
        now = datetime.now()
        self.calls = [call for call in self.calls if now - call < timedelta(hours=1)]
        self.calls.append(now)
        hourly_calls = len(self.calls)
        
        if error:
            self.consecutive_errors += 1
            pause = min(self.default_pause * (self.cooldown_factor ** self.consecutive_errors), self.max_pause)
            logging.warning(f"API error, increasing wait time to {pause:.2f} seconds")
        else:
            self.consecutive_errors = 0
            
            usage_ratio = hourly_calls / self.hourly_limit
            if usage_ratio < 0.5:
                pause = self.default_pause * (0.8 + 0.4 * random.random())
            elif usage_ratio < 0.8:
                pause = self.default_pause * 2 * (0.9 + 0.2 * random.random())
            else:
                pause = self.default_pause * 3 * (0.9 + 0.2 * random.random())
        
        pause = min(pause, self.max_pause)
        
        logging.info(f"API call sleep {pause:.2f} seconds (current usage: {hourly_calls}/{self.hourly_limit})")
        time.sleep(pause)

class SpotifyLyricsFetcher:

    def __init__(self, spotify_client_id, spotify_client_secret, genius_token):
        self.sp = None
        if spotify_client_id and spotify_client_secret:
            try:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=spotify_client_id,
                    client_secret=spotify_client_secret
                )
                self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                logging.info("Spotify API client initialized successfully")
            except Exception as e:
                logging.error(f"Spotify API client initialization failed: {e}")
        
        self.genius = None
        if genius_token:
            try:
                self.genius = lyricsgenius.Genius(genius_token)
                self.genius.verbose = False
                self.genius.remove_section_headers = True
                logging.info("Genius API client initialized successfully")
            except Exception as e:
                logging.error(f"Genius API client initialization failed: {e}")
        
        self.spotify_limiter = SmartRateLimiter(default_pause=1.0, hourly_limit=1000)
        self.genius_limiter = SmartRateLimiter(default_pause=2.0, hourly_limit=100)
        
        self.cache_file = "lyrics_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache file: {e}")
        return {"spotify": {}, "genius": {}}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache file: {e}")
    
    def search_spotify_track(self, artist_name, track_name):
        if not self.sp:
            logging.warning("Spotify client not initialized, skipping Spotify search")
            return None
            
        cache_key = f"{artist_name}|{track_name}".lower()
        if cache_key in self.cache["spotify"]:
            logging.info(f"Getting Spotify info from cache: {artist_name} - {track_name}")
            return self.cache["spotify"][cache_key]
        
        try:
            query = f"artist:{artist_name} track:{track_name}"
            self.spotify_limiter.sleep()
            
            results = self.sp.search(q=query, type="track", limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': track['artists'][0]['name'],
                    'album_name': track['album']['name'],
                    'release_date': track['album']['release_date'],
                    'popularity': track['popularity'],
                    'external_url': track['external_urls']['spotify']
                }
                
                self.cache["spotify"][cache_key] = track_info
                self._save_cache()
                
                return track_info
            else:
                logging.warning(f"Not found in Spotify: {artist_name} - {track_name}")
                return None
                
        except Exception as e:
            logging.error(f"Spotify API error: {e}")
            self.spotify_limiter.sleep(error=True)
            return None

    def get_lyrics(self, artist_name, track_name, use_spotify=True):
        if not self.genius:
            logging.warning("Genius client not initialized, unable to fetch lyrics")
            return None
            
        cache_key = f"{artist_name}|{track_name}".lower()
        if cache_key in self.cache["genius"]:
            logging.info(f"Getting lyrics from cache: {artist_name} - {track_name}")
            return self.cache["genius"][cache_key]
        
        spotify_info = None
        if use_spotify and self.sp:
            spotify_info = self.search_spotify_track(artist_name, track_name)
            if spotify_info:
                artist_name = spotify_info['artist_name']
                track_name = spotify_info['track_name']
                logging.info(f"Spotify identification result: {artist_name} - {track_name}")
        
        try:
            self.genius_limiter.sleep()
            song = self.genius.search_song(track_name, artist_name)
            
            if song:
                lyrics = song.lyrics
                
                self.cache["genius"][cache_key] = lyrics
                self._save_cache()
                
                logging.info(f"Successfully retrieved lyrics: {artist_name} - {track_name}")
                return lyrics
            else:
                logging.warning(f"Lyrics not found: {artist_name} - {track_name}")
                
                self.cache["genius"][cache_key] = None
                self._save_cache()
                
                return None
                
        except Exception as e:
            logging.error(f"Genius API error: {e}")
            self.genius_limiter.sleep(error=True)
            return None
    
    def process_metadata_file(self, input_file, output_file, artist_column='Artist', track_column='Song title'):
        global current_results, current_output_file, current_pbar
        if not os.path.exists(input_file):
            logging.error(f"File does not exist: {input_file}")
            return
        
        try:
            try:
                df = pd.read_csv(input_file, on_bad_lines='skip', engine='python')
            except Exception as e:
                logging.warning(f"Standard parsing method failed: {e}, trying alternative approach...")
                try:
                    df = pd.read_csv(input_file, on_bad_lines='skip', engine='python', 
                                     quoting=3, encoding='utf-8')
                except Exception as e2:
                    logging.warning(f"Alternative 1 failed: {e2}, trying alternative 2...")
                    try:
                        df = pd.read_csv(input_file, quoting=1, escapechar='\\', doublequote=True,
                                         encoding='utf-8', engine='c', on_bad_lines='skip')
                    except Exception as e3:
                        logging.error(f"All CSV parsing methods failed: {e3}")
                        raise
            
            artist_found = False
            track_found = False
            
            if artist_column in df.columns:
                artist_found = True
            else:
                logging.warning(f"Artist column not found: {artist_column}")
                potential_artist_columns = ['Artist', 'artist_name', 'artist_name_normalized', 'artist']
                for col in potential_artist_columns:
                    if col in df.columns:
                        artist_column = col
                        artist_found = True
                        logging.info(f"Using alternative artist column: {artist_column}")
                        break
                    
            if track_column in df.columns:
                track_found = True
            else:
                logging.warning(f"Track column not found: {track_column}")
                potential_track_columns = ['Song title', 'track_name', 'track_name_normalized', 'title', 'song', 'Song title ', 'Track']
                for col in potential_track_columns:
                    if col in df.columns:
                        track_column = col
                        track_found = True
                        logging.info(f"Using alternative track column: {track_column}")
                        break
            
            if not artist_found:
                logging.error("Unable to find artist information column")
                logging.info(f"Available columns: {', '.join(df.columns)}")
                return
                
            if not track_found:
                logging.error("Unable to find track information column")
                logging.info(f"Available columns: {', '.join(df.columns)}")
                return
            
            results = []
            total = len(df)
            
            current_results = results
            current_output_file = output_file
            
            pbar = tqdm(total=total, desc=f"Processing {os.path.basename(input_file)}", 
                        ncols=100, colour="green")
            current_pbar = pbar
            
            for idx, row in df.iterrows():
                artist_name = str(row[artist_column]).strip()
                track_name = str(row[track_column]).strip()
                
                pbar.set_description(f"Processing {artist_name} - {track_name}")
                
                spotify_info = self.search_spotify_track(artist_name, track_name)
                
                lyrics = self.get_lyrics(artist_name, track_name, use_spotify=True)
                
                result = row.to_dict()
                
                if spotify_info:
                    for k, v in spotify_info.items():
                        result[f'spotify_{k}'] = v
                
                if lyrics:
                    try:
                        cleaned_lyrics = lyrics.replace('\r', ' ').replace('\n', ' | ')
                        cleaned_lyrics = re.sub(r'[\x00-\x1F\x7F]', ' ', cleaned_lyrics)
                        result['lyrics'] = cleaned_lyrics
                    except Exception as lyrics_error:
                        logging.warning(f"Error processing lyrics: {lyrics_error}")
                        result['lyrics'] = None
                else:
                    result['lyrics'] = None
                
                results.append(result)
                
                if (idx + 1) % 10 == 0 or (idx + 1) == total:
                    temp_file = output_file + ".temp"
                    pd.DataFrame(results).to_csv(temp_file, index=False, encoding='utf-8', 
                                                  quoting=1, escapechar='\\', doublequote=True)
                    
                    if os.path.exists(output_file):
                        try:
                            os.remove(output_file)
                        except:
                            pass
                    
                    try:
                        os.rename(temp_file, output_file)
                        pbar.set_postfix({"Saved": f"{idx + 1}/{total}"})
                    except Exception as save_error:
                        logging.error(f"Failed to rename temp file: {save_error}")
                
                pbar.update(1)
                
            pbar.close()
            current_pbar = None
            
            results_df = pd.DataFrame(results)
            
            results_df.to_csv(output_file, index=False, encoding='utf-8', quoting=1, escapechar='\\', doublequote=True)
            logging.info(f"Processing completed! Results saved to: {output_file}")
            
            current_results = []
            current_output_file = None
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            
            if current_results and len(current_results) > 0:
                try:
                    error_output = output_file.replace('.csv', '_partial.csv')
                    pd.DataFrame(current_results).to_csv(error_output, index=False, encoding='utf-8', 
                                                  quoting=1, escapechar='\\', doublequote=True)
                    logging.info(f"Saved partial results to {error_output}")
                    print(f"Saved {len(current_results)} partial results to {error_output}")
                except Exception as save_error:
                    logging.error(f"Failed to save partial results: {save_error}")
            
            current_results = []
            current_output_file = None
            if current_pbar:
                current_pbar.close()
                current_pbar = None
                
            return None

current_results = []
current_output_file = None
current_pbar = None

def signal_handler(sig, frame):
    global current_results, current_output_file, current_pbar
    
    if current_pbar:
        current_pbar.close()
    
    if current_results and current_output_file:
        print(f"\nInterrupt received, saving {len(current_results)} results to {current_output_file}")
        pd.DataFrame(current_results).to_csv(current_output_file, index=False, encoding='utf-8', 
                                           quoting=1, escapechar='\\', doublequote=True)
        print(f"Data saved successfully!")
    else:
        print("\nInterrupt received, no data to save")
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    PROCESS_2013 = False
    PROCESS_2014 = True
    PROCESS_2015 = False
    
    PROCESS_ALL_IF_NONE_SELECTED = True
    
    spotify_client_id = os.environ.get('SPOTIFY_CLIENT_ID', '79b4a0434e8f49b18da4b628693d2ad7')
    spotify_client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET', '977d4de5593f4555a4bbcd0452dfc221')
    genius_token = os.environ.get('GENIUS_ACCESS_TOKEN', '5Buz-X0MroxFkiiVWsPBV60PWZpZWlFcCwnNgz2Ed8wnAiwBI2dNrhg2xiSYNXXJ')
    genius_client_secret = os.environ.get('GENIUS_CLIENT_SECRET', 'uaw9vEcMhzM7aAYpUhwSk4lgT3o12xDoS9mRDAnTnfqqw1bjd2CG6bs3mxKeZTLytrxI12x7PFjVDcSDNOvJgA')
    
    spotify_client_id = "79b4a0434e8f49b18da4b628693d2ad7"
    spotify_client_secret = "977d4de5593f4555a4bbcd0452dfc221"
    
    genius_client_id = "5Buz-X0MroxFkiiVWsPBV60PWZpZWlFcCwnNgz2Ed8wnAiwBI2dNrhg2xiSYNXXJ"
    genius_client_secret = "uaw9vEcMhzM7aAYpUhwSk4lgT3o12xDoS9mRDAnTnfqqw1bjd2CG6bs3mxKeZTLytrxI12x7PFjVDcSDNOvJgA"
    
    genius_token = "jt6-BpCkDnNNP-4m5GQj2RxkIOQMhHCW-0LUvx5GwUSRlF6PInq2VAZiPGpjpuWu"
    
    if not all([spotify_client_id, spotify_client_secret, genius_token]):
        logging.error("Set API keys in environment variables or directly in the script")
        return
    
    fetcher = SpotifyLyricsFetcher(
        spotify_client_id=spotify_client_id,
        spotify_client_secret=spotify_client_secret,
        genius_token=genius_token
    )
    
    metadata_dir = "/Users/USERNAME/Desktop/auckland_uni/CompSci_760/msc_emo_pred/datasets/DEAM/metadata"
    output_dir = "/Users/USERNAME/Desktop/auckland_uni/CompSci_760/msc_emo_pred/datasets/processed_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    any_year_selected = PROCESS_2013 or PROCESS_2014 or PROCESS_2015
    
    if not any_year_selected and PROCESS_ALL_IF_NONE_SELECTED:
        files = [f for f in os.listdir(metadata_dir) if f.endswith('.csv') and 'metadata' in f.lower()]
        print(f"\nNo year selected, processing all {len(files)} metadata files\n")
        
        for file in tqdm(files, desc="Overall Progress", ncols=100, colour="blue"):
            input_path = os.path.join(metadata_dir, file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_with_lyrics.csv")
            
            logging.info(f"Starting to process file: {file}")
            fetcher.process_metadata_file(input_path, output_path)
    else:
        files_to_process = []
        
        if PROCESS_2013:
            files_to_process.append("metadata_2013.csv")
        if PROCESS_2014:
            files_to_process.append("metadata_2014_cleaned.csv")
        if PROCESS_2015:
            files_to_process.append("metadata_2015.csv")
            
        print(f"\nProcessing selected {len(files_to_process)} metadata files: {', '.join(files_to_process)}\n")
        
        for target_file in files_to_process:
            input_path = os.path.join(metadata_dir, target_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(target_file)[0]}_with_lyrics.csv")
            
            if not os.path.exists(input_path):
                logging.error(f"File does not exist: {input_path}")
                continue
                
            print(f"Starting to process file: {target_file}")
            logging.info(f"Starting to process file: {target_file}")
            fetcher.process_metadata_file(input_path, output_path)
            print(f"Processing completed! Results saved to: {output_path}")

if __name__ == "__main__":
    main()