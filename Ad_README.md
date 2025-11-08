NOTE: Contains additional info (team specific) apart from general @ README.md

## Setup

1. Clone the team's fork:
```
git clone git@github.com:aadarshram/swarm-rescue.git
```
2. Follow the setup rules at README.md
NOTE: Virtual Env  
    a. Create venv 
    ```
    python3 -m venv .venv (if not exist)
    source .venv/bin/activate
    ```
    b. If requirements.txt exists:
    ```
    python -m pip install -r requirements.txt
    ```

## Making updates
1. Push to a seperate branch not main.
2. Export newly installed packages to requirements.txt
```
python -m pip freeze > requirements.txt
```

## Debugging
1. Check if you are using the correct python and pip under venv
```
which python
which pip
```
2. Write logs to a logfile and use pdb.
3. Whenever possible write pytest functions in test/ dir for testing different modules.

More info will be appended here useful for the developing team.