import requests

# Define the URL of your server
url = 'http://192.168.122.1:7777/refresh_containers'  # Change to your actual endpoint

# Send a GET request
response = requests.get(url)

# Print the response text (the content of the response)
print(response.text)
