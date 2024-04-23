import time

class Lift:
    def __init__(self, num_floors):
        self.num_floors = num_floors
        self.current_floor = 1
        self.direction = 0  # 1 for up, -1 for down, 0 for stop
        self.requests = []

    def move(self):
        if self.direction == 1:
            self.current_floor += 1
        elif self.direction == -1:
            self.current_floor -= 1
        print(f"Moving to floor {self.current_floor}")
        time.sleep(3)  # 3 seconds delay for moving between floors

    def stop(self):
        print(f"Stopping at floor {self.current_floor}")
        self.direction = 0

    def process_requests(self):
        while self.requests:
            request = self.requests.pop(0)
            floor, direction = request
            if floor == self.current_floor:
                self.stop()
            elif floor > self.current_floor:
                self.direction = 1
                while self.current_floor < floor:
                    self.move()
                self.stop()
            else:
                self.direction = -1
                while self.current_floor > floor:
                    self.move()
                self.stop()

    def add_request(self, floor, direction):
        self.requests.append((floor, direction))
        if direction == 1:
            print(f"Request to go UP from floor {floor}")
        elif direction == -1:
            print(f"Request to go DOWN from floor {floor}")

def main():
    num_floors = 10
    lift = Lift(num_floors)
    while True:
        user_input = input("Enter floor number to go (1-10) or 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        try:
            floor = int(user_input)
            if floor < 1 or floor > num_floors:
                print("Invalid floor number. Please enter a number between 1 and 10.")
                continue
            direction = 1 if floor > lift.current_floor else -1
            lift.add_request(floor, direction)
            lift.process_requests()
        except ValueError:
            print("Invalid input. Please enter a valid floor number or 'q' to quit.")

if __name__ == "__main__":
    main()
