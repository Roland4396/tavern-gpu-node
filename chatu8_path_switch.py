"""Compatibility for the author's Path / Input_1..20 media selector.

Only routing is implemented here. An unconnected selected input means no media;
it must not fall back to another input or block the downstream conditioning node.
"""


class PathSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"Path": ("INT", {"default": 1, "min": 1, "max": 20})},
            "optional": {
                f"Input_{index}": ("*", {"lazy": True})
                for index in range(1, 21)
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "select"
    CATEGORY = "utils/routing"
    DESCRIPTION = "Select one of 20 inputs unchanged. An empty selected input returns None."

    @staticmethod
    def _input_name(path):
        if type(path) is not int or not 1 <= path <= 20:
            raise ValueError("Path must be an integer from 1 to 20")
        return f"Input_{path}"

    def check_lazy_status(self, Path, **inputs):
        selected = self._input_name(Path)
        # Comfy passes None for connected-but-unevaluated inputs and omits
        # unconnected optional inputs. Never request an absent connection.
        return [selected] if selected in inputs and inputs[selected] is None else []

    def select(self, Path, **inputs):
        return (inputs.get(self._input_name(Path)),)


NODE_CLASS_MAPPINGS = {"Switch": PathSwitch}
NODE_DISPLAY_NAME_MAPPINGS = {"Switch": "Switch (Path 1–20)"}
