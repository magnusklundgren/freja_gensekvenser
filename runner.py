import subprocess
import time
import sys

def main():
	while True:
		try:
			print("Starting 04_batch.py...")
			# Define arguments directly in the script
			command = [
				sys.executable,
				'-u',
				'04_batch.py',
				'--input_file', 'variants.txt',
				'--batch_size', '200',
				'--workers', '4',
				'--output_file', 'variant_scores.csv',
				'--dna_model', '',
				'--organism', 'human',
				'--sequence_length', '1MB',
				'--chrom_startswith', '1'
			]
			
			result = subprocess.run(command, check=True)
			
			if result.returncode == 0:
				print("04_batch.py completed successfully. Exiting runner.")
				break

		except subprocess.CalledProcessError as e:
			print(f"04_batch.py failed with error: {e}. Restarting in 10 seconds...")
			time.sleep(10)
		except KeyboardInterrupt:
			print("Runner stopped by user.")
			break
		except Exception as e:
			print(f"An unexpected error occurred: {e}. Restarting in 10 seconds...")
			time.sleep(10)

if __name__ == '__main__':
	main()
