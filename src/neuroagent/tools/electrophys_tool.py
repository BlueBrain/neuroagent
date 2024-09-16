"""Electrophys tool."""

import logging
import tempfile
from statistics import mean
from typing import Any, Literal, Optional, Type

from bluepyefe.extract import extract_efeatures
from efel.units import get_unit
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)


POSSIBLE_PROTOCOLS = {
    "idrest": ["idrest"],
    "idthresh": ["idthres", "idthresh"],
    "iv": ["iv"],
    "apwaveform": ["apwaveform"],
    "spontaneous": ["spontaneous"],
    "step": ["step"],
    "spontaps": ["spontaps"],
    "firepattern": ["firepattern"],
    "sponnohold30": ["sponnohold30", "spontnohold30"],
    "sponthold30": ["sponhold30", "sponthold30"],
    "starthold": ["starthold"],
    "startnohold": ["startnohold"],
    "delta": ["delta"],
    "sahp": ["sahp"],
    "idhyperpol": ["idhyeperpol"],
    "irdepol": ["irdepol"],
    "irhyperpol": ["irhyperpol"],
    "iddepol": ["iddepol"],
    "ramp": ["ramp"],
    "apthresh": ["apthresh", "ap_thresh"],
    "hyperdepol": ["hyperdepol"],
    "negcheops": ["negcheops"],
    "poscheops": ["poscheops"],
    "spikerec": ["spikerec"],
    "sinespec": ["sinespec"],
}


STIMULI_TYPES = list[
    Literal[
        "spontaneous",
        "idrest",
        "idthres",
        "apwaveform",
        "iv",
        "step",
        "spontaps",
        "firepattern",
        "sponnohold30",
        "sponhold30",
        "starthold",
        "startnohold",
        "delta",
        "sahp",
        "idhyperpol",
        "irdepol",
        "irhyperpol",
        "iddepol",
        "ramp",
        "ap_thresh",
        "hyperdepol",
        "negcheops",
        "poscheops",
        "spikerec",
        "sinespec",
    ]
]

CALCULATED_FEATURES = list[
    Literal[
        "spike_count",
        "time_to_first_spike",
        "time_to_last_spike",
        "inv_time_to_first_spike",
        "doublet_ISI",
        "inv_first_ISI",
        "ISI_log_slope",
        "ISI_CV",
        "irregularity_index",
        "adaptation_index",
        "mean_frequency",
        "strict_burst_number",
        "strict_burst_mean_freq",
        "spikes_per_burst",
        "AP_height",
        "AP_amplitude",
        "AP1_amp",
        "APlast_amp",
        "AP_duration_half_width",
        "AHP_depth",
        "AHP_time_from_peak",
        "AP_peak_upstroke",
        "AP_peak_downstroke",
        "voltage_base",
        "voltage_after_stim",
        "ohmic_input_resistance_vb_ssse",
        "steady_state_voltage_stimend",
        "sag_amplitude",
        "decay_time_constant_after_stim",
        "depol_block_bool",
    ]
]


class AmplitudeInput(BaseModel):
    """Amplitude class."""

    min_value: float
    max_value: float


class InputElectrophys(BaseModel):
    """Inputs of the NeuroM API."""

    trace_id: str = Field(
        description=(
            "ID of the trace of interest. The trace ID is in the form of an HTTP(S)"
            " link such as 'https://bbp.epfl.ch/neurosciencegraph/data/traces...'."
        )
    )
    stimuli_types: Optional[STIMULI_TYPES] = Field(
        description=(
            "Type of stimuli requested by the user. Should be one of 'spontaneous',"
            " 'idrest', 'idthres', 'apwaveform', 'iv', 'step', 'spontaps',"
            " 'firepattern', 'sponnohold30','sponhold30', 'starthold', 'startnohold',"
            " 'delta', 'sahp', 'idhyperpol', 'irdepol', 'irhyperpol','iddepol', 'ramp',"
            " 'ap_thresh', 'hyperdepol', 'negcheops', 'poscheops',"
            " 'spikerec', 'sinespec'."
        )
    )
    calculated_feature: Optional[CALCULATED_FEATURES] = Field(
        description=(
            "Feature requested by the user. Should be one of 'spike_count',"
            "'time_to_first_spike', 'time_to_last_spike',"
            "'inv_time_to_first_spike', 'doublet_ISI', 'inv_first_ISI',"
            "'ISI_log_slope', 'ISI_CV', 'irregularity_index', 'adaptation_index',"
            "'mean_frequency', 'strict_burst_number', 'strict_burst_mean_freq',"
            "'spikes_per_burst', 'AP_height', 'AP_amplitude', 'AP1_amp', 'APlast_amp',"
            "'AP_duration_half_width', 'AHP_depth', 'AHP_time_from_peak',"
            "'AP_peak_upstroke', 'AP_peak_downstroke', 'voltage_base',"
            "'voltage_after_stim', 'ohmic_input_resistance_vb_ssse',"
            "'steady_state_voltage_stimend', 'sag_amplitude',"
            "'decay_time_constant_after_stim', 'depol_block_bool'"
        )
    )
    amplitude: Optional[AmplitudeInput] = Field(
        description=(
            "Amplitude of the protocol (should be specified in nA)."
            "Can be a range of amplitudes with min and max values"
            "Can be None (if the user does not specify it)"
            " and all the amplitudes are going to be taken into account."
        ),
    )


class FeaturesOutput(BaseToolOutput):
    """Output schema for the neurom tool."""

    brain_region: str
    feature_dict: dict[str, Any]


class ElectrophysFeatureTool(BasicTool):
    """Class defining the Electrophys Featyres Tool."""

    name: str = "electrophys-features-tool"
    description: str = """Given a trace ID, extract features from the trace for certain stimuli types and certain amplitudes.
    You can optionally specify which feature to calculate, for which stimuli types and for which amplitudes:
    - The calculated features are a list of features that the user requests to compute.
    - The stimuli types are the types of input stimuli injected in the cell when measuring the response.
    - The amplitude is the total current injected in the cell when measuring the response.
    Specify those ONLY if the user specified them. Otherwise leave them as None.
    """
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputElectrophys

    def _run(self) -> None:  # type: ignore
        """Not implemented yet."""
        pass

    async def _arun(
        self,
        trace_id: str,
        calculated_feature: CALCULATED_FEATURES | None = None,
        stimuli_types: STIMULI_TYPES | None = None,
        amplitude: AmplitudeInput | None = None,
    ) -> FeaturesOutput | dict[str, str]:
        """Give features about trace.

        Parameters
        ----------
        trace
            ID of the trace of interest (of the form https://bbp.epfl.ch/neurosciencegraph/data/traces...)
        calculated_features
            List of features one wants to compute
        stimuli_types
            List of stimuli types that should be taken into account when computing the features
        amplitude
            Amplitude range of the input stimulus when measuring the cell's response

        Returns
        -------
            Dict of feature: value
        """
        logger.info(
            f"Entering electrophys tool. Inputs: {trace_id=}, {calculated_feature=},"
            f" {amplitude=}, {stimuli_types=}"
        )
        try:
            # Deal with cases where user did not specify stimulus type or/and feature
            if not stimuli_types:
                # Default to IDRest if protocol not specified
                logger.warning("No stimulus type specified. Defaulting to IDRest.")
                stimuli_types = ["idrest"]
            if not calculated_feature:
                # Compute ALL of the available features if not specified
                logger.warning("No feature specified. Defaulting to everything.")
                calculated_feature = list(CALCULATED_FEATURES.__args__[0].__args__)  # type: ignore

            # Download the .nwb file associated to the trace from the KG
            trace_content, metadata = await get_kg_data(
                object_id=trace_id,
                httpx_client=self.metadata["httpx_client"],
                url=self.metadata["url"],
                token=self.metadata["token"],
                preferred_format="nwb",
            )

            # Turn amplitude requirement of user into a bluepyefe compatible representation
            if isinstance(amplitude, AmplitudeInput):
                # If the user specified amplitude/a range of amplitudes,
                # the target amplitude is centered on the range and the
                # tolerance is set as half the range
                desired_amplitude = mean([amplitude.min_value, amplitude.max_value])

                # If the range is just one number, use 10% of it as tolerance
                if amplitude.min_value == amplitude.max_value:
                    desired_tolerance = amplitude.max_value * 0.1
                else:
                    desired_tolerance = amplitude.max_value - desired_amplitude
            else:
                # If the amplitudes are not specified, take an arbitrarily high tolerance
                desired_amplitude = 0
                desired_tolerance = 1e12
            logger.info(
                f"target amplitude set to {desired_amplitude} nA. Tolerance is"
                f" {desired_tolerance} nA"
            )

            targets = []
            # Create a target for each stimuli_types and their various spellings and for each feature to compute
            for stim_type in stimuli_types:
                for efeature in calculated_feature:
                    for protocol in POSSIBLE_PROTOCOLS[stim_type]:
                        target = {
                            "efeature": efeature,
                            "protocol": protocol,
                            "amplitude": desired_amplitude,
                            "tolerance": desired_tolerance,
                        }
                        targets.append(target)
            logger.info(f"Generated {len(targets)} targets.")

            # The trace needs to be opened from a file, no way to hack it
            with (
                tempfile.NamedTemporaryFile(suffix=".nwb") as temp_file,
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                temp_file.write(trace_content)

                # LNMC traces need to be adjusted by an output voltage of 14mV due to their experimental protocol
                if metadata.is_lnmc:
                    files_metadata = {
                        "test": {
                            stim_type: [
                                {
                                    "filepath": temp_file.name,
                                    "protocol": protocol,
                                    "ljp": 14,
                                }
                                for protocol in POSSIBLE_PROTOCOLS[stim_type]
                            ]
                            for stim_type in stimuli_types
                        }
                    }
                else:
                    files_metadata = {
                        "test": {
                            stim_type: [
                                {"filepath": temp_file.name, "protocol": protocol}
                                for protocol in POSSIBLE_PROTOCOLS[stim_type]
                            ]
                            for stim_type in stimuli_types
                        }
                    }

                # Extract the requested features for the requested protocols
                efeatures, protocol_definitions, _ = extract_efeatures(
                    output_directory=temp_dir,
                    files_metadata=files_metadata,
                    targets=targets,
                    absolute_amplitude=True,
                )
                output_features = {}

                # Format the extracted features into a readable dict for the model
                for protocol_name in protocol_definitions.keys():
                    efeatures_values = efeatures[protocol_name]
                    protocol_def = protocol_definitions[protocol_name]
                    output_features[protocol_name] = {
                        f"{f['efeature_name']} (avg on n={f['n']} trace(s))": (
                            f"{f['val'][0]} {get_unit(f['efeature_name']) if get_unit(f['efeature_name']) != 'constant' else ''}".strip()
                        )
                        for f in efeatures_values["soma"]
                    }

                    # Add the stimulus current of the protocol to the output
                    output_features[protocol_name]["stimulus_current"] = (
                        f"{protocol_def['step']['amp']} nA"
                    )
            return FeaturesOutput(
                brain_region=metadata.brain_region, feature_dict=output_features
            )
        except Exception as e:
            raise ToolException(str(e), self.name)
