from src.main import main
# import cProfile
import time

start_time = time.time()


main()

end_time = time.time()

print(f"Code run time: {end_time - start_time:.2f} seconds")

# cProfile.run('main()')
