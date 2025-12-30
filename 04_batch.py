import os
import argparse
import json
import concurrent.futures
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from io import StringIO
import pandas as pd
from tqdm import tqdm


def _atomic_write_json(path: str, data: dict) -> None:
	path_tmp = f"{path}.tmp"
	with open(path_tmp, 'w', encoding='utf-8') as f:
		json.dump(data, f)
		f.flush()
		os.fsync(f.fileno())
	os.replace(path_tmp, path)


def _load_checkpoint(path: str) -> dict | None:
	try:
		with open(path, 'r', encoding='utf-8') as f:
			return json.load(f)
	except FileNotFoundError:
		return None
	except Exception:
		return None


def _tail_last_data_line(path: str, max_bytes: int = 1024 * 1024) -> str | None:
	"""Return last non-empty line from a potentially huge file, without reading it all."""
	try:
		with open(path, 'rb') as f:
			f.seek(0, os.SEEK_END)
			size = f.tell()
			read_size = min(size, max_bytes)
			f.seek(-read_size, os.SEEK_END)
			chunk = f.read(read_size)
		text = chunk.decode('utf-8', errors='ignore')
		lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
		if not lines:
			return None
		return lines[-1]
	except FileNotFoundError:
		return None
	except Exception:
		return None


def _variant_id_to_input_line(variant_id: str) -> str | None:
	# Example variant_id: "chr1:926014:G>A"
	try:
		variant_id = variant_id.strip().strip('"')
		if not variant_id.startswith('chr'):
			return None
		chrom, pos, change = variant_id[3:].split(':', 2)
		ref, alt = change.split('>', 1)
		return f"{chrom}-{pos}-{ref}-{alt}"
	except Exception:
		return None


def _find_variant_index_in_input(input_file: str, target_line: str) -> int | None:
	"""Return 0-based variant index (excluding header) for the matching input line."""
	try:
		with open(input_file, 'r', encoding='utf-8') as f:
			# Skip header
			next(f, None)
			for idx, line in enumerate(f):
				if line.strip() == target_line:
					return idx
		return None
	except Exception:
		return None

def main():
	parser = argparse.ArgumentParser(description="Process variants in batches.")
	parser.add_argument('--input_file', type=str, default='./variants.txt', help='Path to the input file.')
	parser.add_argument('--output_file', type=str, default='variant_scores.csv', help='Path to the output file.')
	parser.add_argument('--batch_size', type=int, default=100, help='Number of variants to process in each batch.')
	parser.add_argument('--commit_every', type=int, default=None, help='How often to flush output + checkpoint (in attempted variants). Defaults to batch_size.')
	parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for score_variant RPCs. Use 1 to disable concurrency.')
	parser.add_argument('--n_sequences', type=int, default=None, help='Maximum number of input variants to process in this run (after resume point).')
	parser.add_argument('--checkpoint_file', type=str, default=None, help='Path to checkpoint JSON file (defaults to <output_file>.checkpoint.json).')
	parser.add_argument('--resume_from', type=int, default=None, help='0-based variant index to resume from (excluding header). Overrides checkpoint/output inference.')
	parser.add_argument('--dna_model', type=str, required=True, help='DNA model to use.')
	parser.add_argument('--organism', type=str, default='human', choices=['human', 'mouse'], help='Organism to use.')
	parser.add_argument('--sequence_length', type=str, default='1MB', choices=["16KB", "100KB", "500KB", "1MB"], help='Length of sequence around variants to predict.')
	parser.add_argument('--column_names', type=str, default='CHROM,POS,REF,ALT', help='Comma-separated list of column names for the input file.')
	parser.add_argument('--chrom_startswith', type=str, default=None, help='Comma-separated list of prefixes to filter by CHROM column (e.g. 1,2,3).')
	
	args = parser.parse_args()

	dna_model = dna_client.create(args.dna_model)

	# --- Checkpoint / Resume Configuration ---
	output_file = args.output_file
	batch_size = args.batch_size
	commit_every = int(args.commit_every) if args.commit_every is not None else int(batch_size)
	if commit_every <= 0:
		commit_every = int(batch_size)
	workers = int(args.workers) if args.workers is not None else 1
	if workers <= 0:
		workers = 1
	checkpoint_file = args.checkpoint_file or f"{output_file}.checkpoint.json"
	resume_source = "unknown"

	# We track resume progress by *input variant index* (0-based, excluding header).
	start_variant_index: int
	if args.resume_from is not None:
		start_variant_index = max(0, int(args.resume_from))
		resume_source = "arg:--resume_from"
		print(f"Resuming from --resume_from={start_variant_index} (0-based, excluding header).")
	else:
		checkpoint = _load_checkpoint(checkpoint_file)
		if checkpoint and isinstance(checkpoint.get('next_variant_index'), int):
			start_variant_index = max(0, int(checkpoint['next_variant_index']))
			resume_source = "checkpoint"
			print(f"Found checkpoint: {checkpoint_file}")
			print(f"Resuming from variant #{start_variant_index} (0-based, excluding header).")
		else:
			# Best-effort inference: if output exists but checkpoint doesn't, infer from the LAST
			# variant_id in the output file (fast tail read) and map it back to the input file.
			start_variant_index = 0
			if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
				print(f"Found existing output file: {output_file}")
				last_line = _tail_last_data_line(output_file)
				last_variant_id = None
				if last_line and not last_line.lower().startswith('variant_id'):
					last_variant_id = last_line.split(',', 1)[0]
					input_line = _variant_id_to_input_line(last_variant_id)
					if input_line:
						idx = _find_variant_index_in_input(args.input_file, input_line)
						if idx is not None:
							start_variant_index = idx + 1
							resume_source = "output:tail-infer"
							print(f"Inferred resume point from output tail: {last_variant_id} -> input index {start_variant_index}")
				if start_variant_index == 0:
					resume_source = "start"
					print("No checkpoint found; starting from the beginning.")
			else:
				resume_source = "start"
				print("Starting from the beginning.")

	# Skip header + already-processed variants
	skip_rows = 1 + start_variant_index

	# Explicit resume banner for quick verification on restarts.
	abspath_input = os.path.abspath(args.input_file)
	abspath_output = os.path.abspath(output_file)
	abspath_checkpoint = os.path.abspath(checkpoint_file)
	output_exists = os.path.exists(output_file)
	output_size = os.path.getsize(output_file) if output_exists else 0
	checkpoint_exists = os.path.exists(checkpoint_file)
	print("\n=== RESUME STATUS ===")
	print(f"resume_source        : {resume_source}")
	print(f"input_file           : {abspath_input}")
	print(f"output_file          : {abspath_output} (exists={output_exists}, bytes={output_size})")
	print(f"checkpoint_file      : {abspath_checkpoint} (exists={checkpoint_exists})")
	print(f"start_variant_index  : {start_variant_index} (0-based, excluding header)")
	print(f"skiprows (pandas)    : {skip_rows} (includes header)")
	print(f"batch_size          : {batch_size}")
	print(f"commit_every        : {commit_every} (attempted variants)")
	print(f"workers             : {workers}")
	print(f"n_sequences         : {args.n_sequences}")
	print("====================\n")

	column_names_list = [c.strip() for c in args.column_names.split(',')]
	if len(column_names_list) != 4:
		print(f"Warning: Expected 4 column names (CHROM, POS, REF, ALT), got {len(column_names_list)}: {column_names_list}")
		# We proceed, but this might fail later if unpacking fails.

	# Create an iterator that reads the file in chunks
	# We manually specify names because we might skip the header row in the file
	vcf_reader = pd.read_csv(
		args.input_file,
		sep='-',
		names=column_names_list,
		header=None,
		skiprows=skip_rows,
		nrows=args.n_sequences,
		chunksize=batch_size,
		dtype={name: 'string' for name in column_names_list},
	)

	chrom_prefixes = None
	if args.chrom_startswith:
		chrom_prefixes = [p.strip() for p in args.chrom_startswith.split(',') if p.strip()]
		print(f"Filtering variants where CHROM starts with: {chrom_prefixes}")

	

	# @markdown Specify length of sequence around variants to predict:
	sequence_length = args.sequence_length
	sequence_length = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
		f'SEQUENCE_LENGTH_{sequence_length}'
	]

	# @markdown Specify which scorers to use to score your variants:
	score_rna_seq = True  # @param { type: "boolean"}
	score_cage = True  # @param { type: "boolean" }
	score_procap = True  # @param { type: "boolean" }
	score_atac = True  # @param { type: "boolean" }
	score_dnase = True  # @param { type: "boolean" }
	score_chip_histone = True  # @param { type: "boolean" }
	score_chip_tf = True  # @param { type: "boolean" }
	score_polyadenylation = True  # @param { type: "boolean" }
	score_splice_sites = True  # @param { type: "boolean" }
	score_splice_site_usage = True  # @param { type: "boolean" }
	score_splice_junctions = True  # @param { type: "boolean" }

	# @markdown Other settings:
	download_predictions = False  # @param { type: "boolean" }

	# Parse organism specification.
	organism_arg = args.organism

	organism_map = {
		'human': dna_client.Organism.HOMO_SAPIENS,
		'mouse': dna_client.Organism.MUS_MUSCULUS,
	}
	organism = organism_map[organism_arg]

	# Parse scorer specification.
	scorer_selections = {
		'rna_seq': score_rna_seq,
		'cage': score_cage,
		'procap': score_procap,
		'atac': score_atac,
		'dnase': score_dnase,
		'chip_histone': score_chip_histone,
		'chip_tf': score_chip_tf,
		'polyadenylation': score_polyadenylation,
		'splice_sites': score_splice_sites,
		'splice_site_usage': score_splice_site_usage,
		'splice_junctions': score_splice_junctions,
	}

	all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
	selected_scorers = [
		all_scorers[key]
		for key in all_scorers
		if scorer_selections.get(key.lower(), False)
	]

	# Remove any scorers or output types that are not supported for the chosen organism.
	unsupported_scorers = [
		scorer
		for scorer in selected_scorers
		if (
			organism.value
			not in variant_scorers.SUPPORTED_ORGANISMS[scorer.base_variant_scorer]
		)
		| (
			(scorer.requested_output == dna_client.OutputType.PROCAP)
			& (organism == dna_client.Organism.MUS_MUSCULUS)
		)
	]
	if len(unsupported_scorers) > 0:
		print(
		  f'Excluding {unsupported_scorers} scorers as they are not supported for'
		  f' {organism}.'
		)
		for unsupported_scorer in unsupported_scorers:
			selected_scorers.remove(unsupported_scorer)


	# --- Processing Loop ---
	# Buffer results and flush output + checkpoint periodically for speed.
	output_has_data = os.path.exists(output_file) and os.path.getsize(output_file) > 0
	next_variant_index = start_variant_index
	last_checkpoint_written_index = start_variant_index
	variant_scores_buffer = []
	buffer_successes = 0
	buffer_attempted_since_last_flush = 0

	global_batch_start = (start_variant_index // batch_size) + 1
	start_offset_in_global_batch = start_variant_index % batch_size
	if start_offset_in_global_batch != 0:
		print(
			f"Note: resume starts mid-batch: global batch {global_batch_start} "
			f"(offset {start_offset_in_global_batch}/{batch_size})."
		)

	def _flush_buffers(reason: str) -> None:
		nonlocal output_has_data, last_checkpoint_written_index, variant_scores_buffer, buffer_successes, buffer_attempted_since_last_flush
		if buffer_successes > 0:
			df_scores = variant_scorers.tidy_scores(variant_scores_buffer)
			if df_scores is not None and len(df_scores) > 0:
				df_scores.to_csv(output_file, mode='a', index=False, header=not output_has_data)
				output_has_data = True
		variant_scores_buffer = []
		buffer_successes = 0
		buffer_attempted_since_last_flush = 0
		# Write checkpoint AFTER output append, so we never skip work that wasn't written.
		last_checkpoint_written_index = next_variant_index
		_atomic_write_json(checkpoint_file, {
			'next_variant_index': next_variant_index,
			'input_file': os.path.abspath(args.input_file),
			'output_file': os.path.abspath(output_file),
			'batch_size': int(batch_size),
			'commit_every': int(commit_every),
			'resume_source': resume_source,
			'flush_reason': reason,
		})

	def _score_one(chrom: str, pos: str, ref: str, alt: str):
		variant = genome.Variant(
			chromosome='chr' + str(chrom),
			position=int(pos),
			reference_bases=str(ref),
			alternate_bases=str(alt),
		)
		interval = variant.reference_interval.resize(sequence_length)
		return dna_model.score_variant(
			interval=interval,
			variant=variant,
			variant_scorers=selected_scorers,
			organism=organism,
		)

	executor = None
	if workers > 1:
		executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
		print(f"Concurrency enabled: {workers} workers")

	try:
		for chunk_index, vcf_chunk in enumerate(vcf_reader):
			global_batch_number = global_batch_start + chunk_index
			attempted_in_chunk = 0
			successes_in_chunk = 0
			rows = list(vcf_chunk.itertuples(index=False, name=None))
			pbar = tqdm(total=len(rows), desc=f"Batch {global_batch_number}")

			for start in range(0, len(rows), commit_every):
				sub = rows[start:start + commit_every]
				attempted = len(sub)
				if attempted == 0:
					continue
				attempted_in_chunk += attempted

				# Filter sub based on chrom_prefixes
				filtered_sub = sub
				if chrom_prefixes:
					filtered_sub = []
					for row in sub:
						# Assuming first element is CHROM (index 0)
						chrom = str(row[0])
						if any(chrom.startswith(p) for p in chrom_prefixes):
							filtered_sub.append(row)
					
					skipped_count = len(sub) - len(filtered_sub)
					if skipped_count > 0:
						pbar.update(skipped_count)

				if executor is None:
					# Sequential fallback
					for chrom, pos, ref, alt in filtered_sub:
						try:
							variant_scores = _score_one(chrom, pos, ref, alt)
							variant_scores_buffer.append(variant_scores)
							buffer_successes += 1
							successes_in_chunk += 1
						except Exception as e:
							print(f"Error processing variant at {chrom}:{pos} - {e}")
						finally:
							pbar.update(1)
				else:
					futures = {}
					for chrom, pos, ref, alt in filtered_sub:
						fut = executor.submit(_score_one, chrom, pos, ref, alt)
						futures[fut] = (chrom, pos)
					for fut in concurrent.futures.as_completed(futures):
						chrom, pos = futures[fut]
						try:
							variant_scores = fut.result()
							variant_scores_buffer.append(variant_scores)
							buffer_successes += 1
							successes_in_chunk += 1
						except Exception as e:
							print(f"Error processing variant at {chrom}:{pos} - {e}")
						finally:
							pbar.update(1)

				buffer_attempted_since_last_flush += attempted
				next_variant_index += attempted
				_flush_buffers(reason=f"threshold:{commit_every}")

			pbar.close()

			# End-of-chunk: flush so we always checkpoint at batch boundaries.
			if buffer_attempted_since_last_flush > 0 or buffer_successes > 0:
				_flush_buffers(reason=f"end_of_batch:{global_batch_number}")
			print(
				f"Batch {global_batch_number} done. Attempted={attempted_in_chunk}, successes={successes_in_chunk}. "
				f"Checkpoint next_variant_index={next_variant_index}"
			)

		# Final flush safety (should be no-op due to end_of_batch flush).
		if next_variant_index != last_checkpoint_written_index:
			_flush_buffers(reason="final")
	finally:
		if executor is not None:
			executor.shutdown(wait=True)

	print("Processing complete.")

if __name__ == '__main__':
	main()
