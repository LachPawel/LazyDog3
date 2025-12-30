import robot
import time

print("--- Robot Movement Test ---")

print("1. Moving FORWARD for 1 second...")
robot.forward()
time.sleep(1)
robot.stopFB()
print("Stopped.")

time.sleep(1)

print("2. Moving BACKWARD for 1 second...")
robot.backward()
time.sleep(1)
robot.stopFB()
print("Stopped.")

time.sleep(1)

print("3. Turning LEFT for 1 second...")
robot.left()
time.sleep(1)
robot.stopLR()
print("Stopped.")

time.sleep(1)

print("4. Turning RIGHT for 1 second...")
robot.right()
time.sleep(1)
robot.stopLR()
print("Stopped.")

print("--- Test Complete ---")
