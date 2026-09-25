import random
from tkinter import Button, Entry, Label, Tk, Toplevel, messagebox

from filters import images
from filters import image_io
from filters import utility


def ask_guess():
    root = Tk()
    root.withdraw()

    dialog = Toplevel(root)
    dialog.title("Image Guess")
    dialog.geometry("560x180")
    dialog.configure(bg="black")
    dialog.attributes("-topmost", True)

    # Small popup for the mini image-guess game
    label = Label(
        dialog,
        text="What image is this?",
        font=("Arial", 24, "bold"),
        fg="white",
        bg="black",
        pady=18,
    )
    label.pack(fill="both", expand=True)

    answer = {"value": ""}

    def submit():
        answer["value"] = entry.get().strip()
        dialog.destroy()
        root.destroy()

    entry = Entry(dialog, font=("Arial", 20, "bold"), width=30)
    entry.pack(pady=(0, 10))
    entry.focus_set()

    button = Button(dialog, text="Check", font=("Arial", 16, "bold"), command=submit)
    button.pack()

    dialog.bind("<Return>", lambda event: submit())
    dialog.wait_window(dialog)
    return answer["value"]


def show_result(message):
    result_root = Tk()
    result_root.withdraw()
    result_root.after(2500, result_root.destroy)
    messagebox.showinfo("Result", message, master=result_root)
    result_root.mainloop()


def run():
    image_directory = "filters/../images"

    filename = utility.random_image(image_directory)
    file = filename.rsplit(".", 1)[0]
    filename = f"{image_directory}/{filename}"
    img = image_io.read_image(filename)

    available_filters = [
        ("greyscale", "numpy", {}),
        ("pixelator", "numpy", {"blocksize": 40}),
        ("ascii", "numpy", {"scale": 2}),
        ("sepia", "numpy", {}),
    ]

    number_of_filters = random.randint(1, len(available_filters))
    filters = random.sample(available_filters, number_of_filters)

    for filter_name, implementation, filter_args in filters:
        filter_fn = images.get_filter(filter_name, implementation)
        img, _ = utility.time_function(filter_fn, img, **filter_args)

    # Show transformed image before guessing
    image_io.display(img)
    guess = ask_guess()

    # give the image a moment to close automatically after the result is shown
    if guess and guess.lower() == file.lower():
        show_result("Correct!")
    else:
        show_result(f"Wrong. The correct answer was: {file}")


if __name__ == "__main__":
    run()
