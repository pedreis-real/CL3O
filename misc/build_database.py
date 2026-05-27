# ================================================================================
# DATABASE API - Build database for required analysis
# ================================================================================

class BuildDatabase:
    '''
    Organises and persists all user-defined input data to disk as JSON files.

    The class is intentionally stateless — every method is a @staticmethod
    so it can be called without creating an instance.

    The class is called only once per runtime to build up missing data.
    '''
    def __init__(self) -> None:
        pass
    
    # ----------------------------------------------------------------
    # 1 - Wing geometry
    # ----------------------------------------------------------------

    @staticmethod
    def create_wng_db(
        wing_specs: dict[str, Any],
        db_filepath: str | Path,
        enbl_log: bool = True,
    ) -> None:
        '''
        Instantiate and persists the wing (and any LS)
        external / planform data.

        All dimensional quantities are in [mm] or [degree].
        Array-like variables must have same size.
        '''
        MainHelpers.banner("BuildDatabase > create_wng_db")
        
        Wing(
            wing_specs     = wing_specs,
            db_filepath    = db_filepath,
            enable_logging = enbl_log,
        )
    
    # ----------------------------------------------------------------
    # 2 - Airfoil adimensional points
    # ----------------------------------------------------------------

    @staticmethod
    def create_afl_db(
        afl_name: str,
        db_filepath: str | Path,
        enbl_log: bool = True,
    ) -> None:
        '''
        Load airfoil coordinates from a .dat file and save processed
        data as a JSON archive.

        Expects .dat airfoil file in Selig format (up TE -> LE low -> TE)
        '''
        MainHelpers.banner("BuildDatabase > create_afl_db")

        Airfoil(
            filename       = f"{afl_name}",
            db_filepath    = db_filepath,
            enable_logging = enbl_log,
        )

    # ----------------------------------------------------------------
    # 3 - Laminate - Index notation
    # ----------------------------------------------------------------

    @staticmethod
    def create_mat_db(
        lam_specs: dict[str, Any],
        db_filepath: str | Path,
        enbl_log: bool = True,
    ) -> None:
        '''
        Alternative for building LainateData.

        Prefer to use ::materials.composite_library::
        '''
        MainHelpers.banner("BuildDatabase > create_mat_db")

        lam = Laminate(
            name           = lam_specs["name"],
            db_filepath    = db_filepath,
            enable_logging = enbl_log,
        )
        for ply in lam_specs["plies"]:
            lam.add_ply(**ply)
        lam.define_laminate_data()

    # ----------------------------------------------------------------
    # 4 - Operational points - V-n envelope and atmospheric data
    # ----------------------------------------------------------------

    @staticmethod
    def create_opp_db(
        opp_specs: dict[str, Any],
        db_filepath: str | Path,
        enbl_log: bool = True,
    ) -> None:
        '''
        Persist V-n envelope / atmospheric conditions set, saved as OppData.

        opp_specs = {
            "aircraft_name" : str,
            "conditions"    : dict,         # keyed by conditions_tag, see
                                            # OperationalPoints docstring
            "input_units"   : list[str]     # optional, default ["m/s", "m"]
        }
        '''
        MainHelpers.banner("BuildDatabase > create_opp_db")

        OperationalPoints(
            aircraft_name  = opp_specs["aircraft_name"],
            conditions     = opp_specs["conditions"],
            db_filepath    = db_filepath,
            input_units    = opp_specs.get("input_units", ["m/s", "m"]),
            enable_logging = enbl_log,
        )

    # ----------------------------------------------------------------
    # 5 - External and 6 - Internal loads from XFLR5
    # ----------------------------------------------------------------

    @staticmethod
    def create_lds_db(
        lds_specs: dict[str, Any],
        db_filepath: str | Path,
        enbl_log: bool = True,
    ) -> None:
        '''
        Distribute and persist XFLR5 external loads as ExLoadsData and
        integrate them into InLoadsData.

        exl_specs = {
            "aircraft_name" : str,
            "conditions"    : list[str],     # condition tags to process
            "xflr5_files"   : list[str],     # raw XFLR5 .txt filenames
        }
        '''
        MainHelpers.banner("BuildDatabase > create_lds_db")

        LoadMapper(
            aircraft_name  = lds_specs.get("aircraft_name", None),
            db_filepath    = db_filepath,
            conditions     = lds_specs.get("conditions",  None),
            xflr5_files    = lds_specs.get("xflr5_files", None),
            enable_logging = enbl_log,
        )
