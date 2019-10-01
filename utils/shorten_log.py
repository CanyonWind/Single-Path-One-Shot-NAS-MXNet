def main(file_name, output_file):
    with open(file_name, 'r') as f:
        content = f.readlines()

    count = 1
    with open(output_file, 'w') as f:
        for line in content:
            if line[0] == '[':
                f.write(line)
                if count % 3 == 0:
                    f.write('-' * 40 + '\n')
                count += 1


if __name__ == '__main__':
    main('../shufflenas_supernet.log', '../condensed_shufflenas_supernet.log')
