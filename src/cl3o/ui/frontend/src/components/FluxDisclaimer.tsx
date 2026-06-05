// Right-side view of the wing: airfoil silhouette with the two spars
// dividing it into the three closed cells (I = LE→aft spar, II = aft→rear,
// III = rear→TE). Arrows trace the CCW positive flow on each cell. Shown
// under the Stress sidebar when the Flux mode is active.
export function FluxDisclaimer() {
  return (
    <div className="flux-disclaimer">
      <p className="disclaimer-text">
        Positive shear flow <b>q &gt; 0</b> follows the
        counter-clockwise (CCW) convention for each closed cell,
        consistent with the Megson idealisation used in CL3O.
        Right-side view of the wing below.
      </p>
      <svg viewBox="0 0 240 110" className="flux-svg" aria-label="right view of wing showing 3-cell flux convention">
        <defs>
          <marker id="arr-b" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#4f8cff"/>
          </marker>
          <marker id="arr-g" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#26a76e"/>
          </marker>
          <marker id="arr-y" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#e6ce00"/>
          </marker>
        </defs>

        {/* Airfoil silhouette (right view of wing). Cambered NACA-like outline
            with LE at x≈20, TE at x≈220; spars at x≈90 (aft) and x≈150 (rear). */}
        <path
          d="M20,55
             C 35,30  80,22  130,28
             C 175,33 205,42 220,55
             C 205,68 175,72  130,75
             C 80,75  35,68  20,55 Z"
          fill="rgba(255,255,255,0.04)"
          stroke="#7c8aa5"
          strokeWidth="1.4"
        />

        {/* Aft and rear spars (vertical webs). */}
        <line x1="90"  y1="30" x2="90"  y2="76" stroke="#7c8aa5" strokeWidth="1" strokeDasharray="3 2"/>
        <line x1="150" y1="29" x2="150" y2="76" stroke="#7c8aa5" strokeWidth="1" strokeDasharray="3 2"/>

        {/* Cell labels. */}
        <text x="55"  y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">I</text>
        <text x="120" y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">II</text>
        <text x="185" y="55" textAnchor="middle" fontSize="11" fill="#c9d4e3">III</text>

        {/* CCW arrows: top going LE-ward (left), bottom going TE-ward (right). */}
        {/* Cell I (LE → aft spar) */}
        <path d="M80,34 L40,34"  stroke="#4f8cff" strokeWidth="1.5" markerEnd="url(#arr-b)" fill="none"/>
        <path d="M35,72 L75,72"  stroke="#4f8cff" strokeWidth="1.5" markerEnd="url(#arr-b)" fill="none"/>
        {/* Cell II (aft → rear spar) */}
        <path d="M140,32 L100,32" stroke="#26a76e" strokeWidth="1.5" markerEnd="url(#arr-g)" fill="none"/>
        <path d="M95,74  L135,74" stroke="#26a76e" strokeWidth="1.5" markerEnd="url(#arr-g)" fill="none"/>
        {/* Cell III (rear spar → TE) */}
        <path d="M205,38 L160,38" stroke="#e6ce00" strokeWidth="1.5" markerEnd="url(#arr-y)" fill="none"/>
        <path d="M155,72 L200,72" stroke="#e6ce00" strokeWidth="1.5" markerEnd="url(#arr-y)" fill="none"/>

        {/* Axis hint (chord arrow + label). */}
        <line x1="20" y1="100" x2="220" y2="100" stroke="#3a4660" strokeWidth="0.8" markerEnd="url(#arr-b)"/>
        <text x="120" y="108" textAnchor="middle" fontSize="8" fill="#7c8aa5">x (chord) — right view of wing  ↺ CCW</text>
      </svg>
    </div>
  );
}
