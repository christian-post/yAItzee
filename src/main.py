import random



def create_full_house_lookup() -> set:
    full_houses = set()
    for i in range(1, 7):
        for j in range(1, 7):
            if i == j: continue

            fh = [str(i)] * 2 + [str(j)] * 3
            full_houses.add("".join(fh))

    return full_houses


def is_full_house(sequence: str, lookup_table: set) -> bool:
    return sequence in lookup_table


def initial_roll() -> str:
    # sort string to reduce the number of possible values
    return "".join(sorted(random.choices("123456", k=5)))


if __name__ == "__main__":
    full_houses = create_full_house_lookup()

    num_rolls = 100000
    full_house_count = 0

    for n in range(num_rolls):
        roll = initial_roll()

        if is_full_house(roll, full_houses):
            full_house_count += 1

    print(round(full_house_count / num_rolls * 100, 2))

