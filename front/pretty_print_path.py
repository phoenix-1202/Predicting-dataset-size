import os


def shorten_path(path: str, max_length: int) -> str:
    if len(path) < max_length:
        return path
    
    shortened_path = "..."
    paths_to_choose_from = path.split(os.sep)
    
    add_last_path = True
    while len(shortened_path) < max_length:
        if len(paths_to_choose_from) == 0:
            return shortened_path
        
        if add_last_path:
            shortened_path = shortened_path.replace("...", f"...{os.sep}{paths_to_choose_from[-1]}")
            del paths_to_choose_from[-1]
            add_last_path = False
        else:
            shortened_path = shortened_path.replace("...", f"{paths_to_choose_from[0]}{os.sep}...")
            del paths_to_choose_from[0]
            add_last_path = True
    return shortened_path
