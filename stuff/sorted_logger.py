
class SortedLogger:
    def __init__(self, log_path, header):
        self.log_path = log_path
        self.header = header
        self.strs = []
        self.footer = ""
        self.__write_to_disk()

    def add_str(self, sort_value, str):
        self.strs.append((sort_value, str))

        # Sort self.metrics by sort_value in descending order
        def sort_func(tuple): return tuple[0]
        self.strs.sort(reverse=True, key=sort_func)

        self.__write_to_disk()

    def set_footer(self, str):
        self.footer = str
        self.__write_to_disk()

    def __write_to_disk(self):
        # Write the header & metrics to log_paths. If log_paths already exists this will overwrite it.
        # This is done so logging tools like 'watch head -n 26 <path to metrics file>' can display
        # live changes to the file in order to track the sorted metrics
        with open(self.log_path, 'w') as fp:
            fp.write(f"{self.header}")
            [fp.write(f"{str}\n") for _, str in self.strs]
            fp.write(f"{self.footer}")
