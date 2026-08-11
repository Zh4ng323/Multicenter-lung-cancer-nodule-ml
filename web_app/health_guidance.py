# -*- coding: utf-8 -*-
"""
Risk-stratified screening and health guidance HTML generator.
Compact single-line layout; disclaimer folded into title bar.
"""

from risk_stratifier import RISK_COLORS

_GUIDANCE = {
    'low-risk': """
      <div style="margin:0 0 5px 0; font-size:15px; font-weight:700;">🟢 Low Risk — Baseline Prevention</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Screening:</b> LDCT every 3 years; annual PFT + tumor markers</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Lifestyle:</b> Avoid tobacco/fumes; regular exercise; light diet; adequate sleep</div>
      <div style="margin:2px 0; line-height:1.5;"><b>See Doctor:</b> Persistent cough &gt; 2 weeks → respiratory clinic</div>
    """,

    'medium-risk': """
      <div style="margin:0 0 5px 0; font-size:15px; font-weight:700;">🟡 Medium Risk — Annual Monitoring</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Screening:</b> Annual LDCT (same site); annual PFT + full tumor markers</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Lifestyle:</b> Quit smoking; N95 on smoggy days; breathing exercise; limit alcohol</div>
      <div style="margin:2px 0; line-height:1.5;"><b>See Doctor:</b> Cough &gt; 3 wks, bloody sputum, weight loss → early LDCT</div>
    """,

    'high-risk': """
      <div style="margin:0 0 5px 0; font-size:15px; font-weight:700;">🔴 High Risk — Intensive Follow-up</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Screening:</b> Thin-slice LDCT q6mo; semi-annual markers; thoracic specialist</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Lifestyle:</b> Permanent cessation (incl. e-cig); gentle exercise; flu + pneumococcal vax</div>
      <div style="margin:2px 0; line-height:1.5;"><b>Red-Flag:</b> Hemoptysis, chest pain, dyspnea, rapid weight loss → emergency care</div>
    """,
}

_PLACEHOLDER = (
    '<div style="color:#999; font-size:15px; text-align:center; padding:10px;">'
    'Health guidance will appear after prediction...</div>'
)


def get_guidance_html(risk_level: str) -> str:
    """Return styled HTML guidance for the given risk level."""
    if not risk_level or risk_level not in _GUIDANCE:
        return _PLACEHOLDER

    color = RISK_COLORS.get(risk_level, '#999')
    content = _GUIDANCE[risk_level]

    # Title + "Reference only" on one line (one size larger for screenshots)
    return (
        f'<div style="border:1px solid #d0d7de; border-radius:6px; padding:8px 10px; '
        f'margin-top:2px; font-size:14px; line-height:1.45; background:#fafbfc;">'
        f'<div style="background:{color}; color:white; padding:4px 10px; border-radius:4px; '
        f'font-size:15px; font-weight:700; margin-bottom:6px; letter-spacing:0.2px; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'Screening &amp; Health Guidance'
        f'<span style="font-weight:400; font-size:12px; opacity:0.9; margin-left:8px;">'
        f'Reference only</span>'
        f'</div>'
        f'{content}'
        f'</div>'
    )
